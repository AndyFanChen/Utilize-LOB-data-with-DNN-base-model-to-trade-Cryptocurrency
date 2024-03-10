# load packages

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, \
    ConfusionMatrixDisplay
import time
import torch
from torch.utils import data
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import argparse
import math
import os


class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, x, y, num_classes, dim):
        """Initialization"""
        self.num_classes = num_classes
        self.dim = dim
        self.x = x
        self.y = y

        self.length = x.shape[0] - self.dim
        # self.length = x.shape[0] - T - self.dim + 1
        print(f"x len {self.length}")

        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x, 1)
        self.y = torch.from_numpy(y)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, i):
        input = self.x[i:i + self.dim, :]
        input = input.permute(1, 0, 2)

        return input, self.y[i]


def _conv1d1x1(in_channels, out_channels):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                         nn.BatchNorm1d(out_channels))


# class taken from https://github.com/jeya-maria-jose/Medical-Transformer/blob/main/lib/models/axialnet.py
class GatedAxialAttention(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dim, flag):
        assert (in_channels % heads == 0) and (out_channels % heads == 0)
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dim_head_v = out_channels // heads
        self.flag = flag  # if flag then we do the attention along width
        self.dim = dim
        self.dim_head_qk = self.dim_head_v // 2
        self.qkv_channels = self.dim_head_v + self.dim_head_qk * 2

        # Multi-head self attention
        self.to_qkv = _conv1d1x1(in_channels, self.heads * self.qkv_channels)
        self.bn_qkv = nn.BatchNorm1d(self.heads * self.qkv_channels)
        self.bn_similarity = nn.BatchNorm2d(heads * 3)
        self.bn_output = nn.BatchNorm1d(self.heads * self.qkv_channels)

        # Gating mechanism
        self.f_qr = nn.Parameter(torch.tensor(0.3), requires_grad=False)
        self.f_kr = nn.Parameter(torch.tensor(0.3), requires_grad=False)
        self.f_sve = nn.Parameter(torch.tensor(0.3), requires_grad=False)
        self.f_sv = nn.Parameter(torch.tensor(0.5), requires_grad=False)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.dim_head_v * 2, dim * 2 - 1), requires_grad=True)
        query_index = torch.arange(dim).unsqueeze(0)
        key_index = torch.arange(dim).unsqueeze(1)
        relative_index = key_index - query_index + dim - 1
        self.register_buffer('flatten_index', relative_index.view(-1))

        self.reset_parameters()
        # self.print_para()

    def forward(self, x):

        if self.flag:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        x = self.to_qkv(x)

        qkv = self.bn_qkv(x)
        q, k, v = torch.split(qkv.reshape(N * W, self.heads, self.dim_head_v * 2, H),
                              [self.dim_head_v // 2, self.dim_head_v // 2, self.dim_head_v], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.dim_head_v * 2, self.dim,
                                                                                       self.dim)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.dim_head_qk, self.dim_head_qk, self.dim_head_v],
                                                            dim=0)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        # multiply by factors
        qr = torch.mul(qr, self.f_qr)
        kr = torch.mul(kr, self.f_kr)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.heads, H, H).sum(dim=1)
        # stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, heads, H, H, W)
        similarity = torch.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

        # multiply by factors
        sv = torch.mul(sv, self.f_sv)
        sve = torch.mul(sve, self.f_sve)

        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_channels * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_channels, 2, H).sum(dim=-2)

        if self.flag:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        return output

    def reset_parameters(self):
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.dim_head_v))


class AxialLOB(nn.Module):
    def __init__(self, W, H, c_in, c_out, c_final, n_heads, pool_kernel, pool_stride, pool_size):
        super().__init__()

        """
        Args:
          W and H:  the width and height of the input tensors
          c_in, c_out, and c_final:  the number of channels for the input, intermediate, and final convolutional layers
          n_heads:  the number of heads for the multi-head attention mechanism used in the GatedAxialAttention layers.
          pool_kernel and pool_stride:  the kernel size and stride of the final average pooling layer.
        """

        # channel output of the CNN_in is the channel input for the axial layer
        self.c_in = c_in
        self.c_out = c_out
        self.c_final = c_final

        self.CNN_in = nn.Conv2d(in_channels=1, out_channels=c_in, kernel_size=1)
        self.CNN_out = nn.Conv2d(in_channels=c_out, out_channels=c_final, kernel_size=1)
        self.CNN_res2 = nn.Conv2d(in_channels=c_out, out_channels=c_final, kernel_size=1)
        self.CNN_res1 = nn.Conv2d(in_channels=1, out_channels=c_out, kernel_size=1)

        self.norm = nn.BatchNorm2d(c_in)
        self.res_norm2 = nn.BatchNorm2d(c_final)
        self.res_norm1 = nn.BatchNorm2d(c_out)
        self.norm2 = nn.BatchNorm2d(c_final)
        self.axial_height_1 = GatedAxialAttention(c_out, c_out, n_heads, H, flag=False)
        self.axial_width_1 = GatedAxialAttention(c_out, c_out, n_heads, W, flag=True)
        self.axial_height_2 = GatedAxialAttention(c_out, c_out, n_heads, H, flag=False)
        self.axial_width_2 = GatedAxialAttention(c_out, c_out, n_heads, W, flag=True)

        self.activation = nn.ReLU()
        # self.linear = nn.Linear(1600, 3)
        self.linear = nn.Linear(n_heads * H * W // pool_size, 3)
        self.pooling = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x):
        # up branch
        # first convolution before the attention
        y = self.CNN_in(x)
        y = self.norm(y)
        y = self.activation(y)

        # attention mechanism through gated multi head axial layer
        y = self.axial_width_1(y)
        y = self.axial_height_1(y)

        # lower branch
        x = self.CNN_res1(x)
        x = self.res_norm1(x)
        x = self.activation(x)

        # first residual
        y = y + x
        z = y.detach().clone()

        # second axial layer
        y = self.axial_width_2(y)
        y = self.axial_height_2(y)

        # second convolution
        y = self.CNN_out(y)
        y = self.res_norm2(y)
        y = self.activation(y)

        # lower branch
        z = self.CNN_res2(z)
        z = self.norm2(z)
        z = self.activation(z)

        # second res connection
        y = y + z

        # final part
        y = self.pooling(y)
        y = torch.flatten(y, 1)
        y = self.linear(y)
        forecast_y = torch.softmax(y, dim=1)
        return forecast_y


def parse_args():

    parser = argparse.ArgumentParser(description='Process file paths.')

    parser.add_argument('--data_path', type=str, required=True,
                        help='Base path to the dataset directory.')
    parser.add_argument('--train_file', type=str, required=True,
                        help='Filename for the training dataset.')
    parser.add_argument('--valid_file', type=str, required=True,
                        help='Filename for the validation dataset.')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Filename for the test dataset.')
    parser.add_argument('--device', type=str, required=True,
                        help='Device for training')
    args = parser.parse_args()
    return args


def batch_gd(model, criterion, optimizer, lr_scheduler, epochs):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    best_test_loss = np.inf
    best_test_epoch = 0
    i = 0
    cont = 0
    for it in tqdm(range(epochs)):

        if (it == 4):
            model.axial_height_1.f_qr.requires_grad = True
            model.axial_height_1.f_kr.requires_grad = True
            model.axial_height_1.f_sve.requires_grad = True
            model.axial_height_1.f_sv.requires_grad = True

            model.axial_width_1.f_qr.requires_grad = True
            model.axial_width_1.f_kr.requires_grad = True
            model.axial_width_1.f_sve.requires_grad = True
            model.axial_width_1.f_sv.requires_grad = True

            model.axial_height_2.f_qr.requires_grad = True
            model.axial_height_2.f_kr.requires_grad = True
            model.axial_height_2.f_sve.requires_grad = True
            model.axial_height_2.f_sv.requires_grad = True

            model.axial_width_2.f_qr.requires_grad = True
            model.axial_width_2.f_kr.requires_grad = True
            model.axial_width_2.f_sve.requires_grad = True
            model.axial_width_2.f_sv.requires_grad = True

        model.train()
        t0 = datetime.now()
        train_loss = []
        train_bar = tqdm(train_loader, desc='train loader')
        # step = 0
        for inputs, targets in train_bar:
            # step += 1
            # move data to GPU
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            train_loss.append(loss.item())
            train_bar.set_postfix(Learning_Rate=optimizer.param_groups[0]['lr'], Training_Loss=train_loss[-1])

        # Get train loss and test loss
        train_loss = np.mean(train_loss)

        model.eval()
        test_loss = []


        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())

        test_loss = np.mean(test_loss)

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss
        lr_scheduler.step()

        # We save the best model
        if test_loss < best_test_loss:
            torch.save(model, os.path.join(data_path, "best_model"))
            best_test_loss = test_loss
            best_test_epoch = it
            print('model saved')

        dt = datetime.now() - t0

        print(f'Epoch {it + 1}/{epochs}, Train Loss: {train_loss:.4f}, \
          Validation Loss: {test_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch}')

    return train_losses, test_losses


# =============================================================================
# # please change the data_path to your local path and unzip the file
if __name__ == "__main__":
    args = parse_args()

    data_path = args.data_path
    train_data_path = os.path.join(args.data_path, args.train_file)
    valid_data_path = os.path.join(args.data_path, args.valid_file)
    test_data_path = os.path.join(args.data_path, args.test_file)
    device = args.device

    dec_data = pd.read_csv(train_data_path)

    dec_val_all = np.genfromtxt(valid_data_path, delimiter=',', skip_header=1)
    dec_test_all = np.genfromtxt(test_data_path, delimiter=',', skip_header=1)
    dec_train_all = np.genfromtxt(train_data_path, delimiter=',', skip_header=1)

    dec_train = dec_train_all[:, :40]
    dec_train_label = dec_train_all[:, -1]

    dec_val = dec_val_all[:, :40]
    dec_val_label = dec_val_all[:, -1]

    dec_test = dec_test_all[:, :40]
    dec_test_label = dec_test_all[:, -1]

    W = 40  # number of features
    dim = 100  # number of LOB states


    y_train = dec_train_label.flatten()
    y_val = dec_val_label.flatten()
    y_test = dec_test_label.flatten()

    # Hyperparameters

    batch_size = 64
    epochs = 50
    c_final = 4  # channel output size of the second conv
    n_heads = 4
    c_in_axial = 32  # channel output size of the first conv
    c_out_axial = 32
    pool_size = 4
    pool_kernel = (1, pool_size)
    pool_stride = (1, pool_size)

    num_classes = 3

    dataset_val = Dataset(dec_val, y_val, num_classes, dim)
    dataset_test = Dataset(dec_test, y_test, num_classes, dim)
    dataset_train = Dataset(dec_train, y_train, num_classes, dim)

    print(dec_val.shape)
    print(y_val.shape)

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

    model = AxialLOB(W, dim, c_in_axial, c_out_axial, c_final, n_heads, pool_kernel, pool_stride, pool_size)
    model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00001)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("------- List Hyper Parameters -------")
    print("epochs   ->   " + str(epochs))
    print("batch size   ->    " + str(batch_size))
    print("Optimizer   ->    " + str(optimizer))
    print(f"Total number of trainable parameters: {trainable_params}")

    train_losses, val_losses = batch_gd(model, criterion, optimizer, scheduler, epochs)

    model = torch.load(os.path.join(data_path, "best_model_axial"))

    n_correct = 0.
    n_total = 0.
    all_targets = []
    all_predictions = []

    for inputs, targets in tqdm(test_loader, desc="test"):
        # Move to GPU
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

        # Forward pass
        outputs = model(inputs)

        # Get prediction
        # torch.max returns both max and argmax
        _, predictions = torch.max(outputs, 1)

        # update counts
        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    test_acc = n_correct / n_total
    print(f"Test acc: {test_acc:.4f}")

    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)

    # print('accuracy_score:', accuracy_score(all_targets, all_predictions))
    print(classification_report(all_targets, all_predictions, digits=4))
    # from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

    c = confusion_matrix(all_targets, all_predictions, normalize="true")
    disp = ConfusionMatrixDisplay(c)
    disp.plot()
    plt.savefig(os.path.join(data_path, "confution_matrix_axial.png"), dpi=300)

    # 計算評估指標
    accuracy = accuracy_score(all_targets, all_predictions) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted')
    precision *= 100
    recall *= 100
    f1 *= 100

    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Precision: {precision:.2f}%')
    print(f'Recall: {recall:.2f}%')
    print(f'F1 Score: {f1:.2f}%')


