# load packages
import pandas as pd
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import os
import torch
import torch.nn.functional as F
from torch.utils import data
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns

def parse_args():
    """
    Parses command line arguments.

    Returns:
        Namespace: An argparse.Namespace class instance containing parsed arguments.
    """
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


def prepare_x(data):
    df1 = data[:, :-1]
    return np.array(df1)


def get_label(data):
    lob = data[:, -1]
    return lob


def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX, dataY


def torch_data(x, y):
    x = torch.from_numpy(x)
    x = torch.unsqueeze(x, 1)
    y = torch.from_numpy(y)
    y = F.one_hot(y, num_classes=3)
    return x, y


class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, data, num_classes, T):
        """Initialization"""

        self.num_classes = num_classes
        self.T = T

        x = prepare_x(data)
        y = get_label(data)
        x, y = data_classification(x, y, self.T)
        self.length = len(x)

        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x, 1)
        self.y = torch.from_numpy(y)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]


class deeplob(nn.Module):
    def __init__(self, y_len):
        super().__init__()
        self.y_len = y_len

        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # lstm layers
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, self.y_len)

    def forward(self, x):
        # h0: (number of hidden layers, batch size, hidden size)
        h0 = torch.zeros(1, x.size(0), 64).to(device)
        c0 = torch.zeros(1, x.size(0), 64).to(device)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)

        #         x = torch.transpose(x, 1, 2)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))

        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc1(x)
        forecast_y = torch.softmax(x, dim=1)

        return forecast_y


def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    best_test_loss = np.inf
    best_test_epoch = 0

    for it in tqdm(range(epochs)):

        model.train()
        t0 = datetime.now()
        train_loss = []
        step = 0
        for inputs, targets in tqdm(train_loader, desc="Training"):
            step += 1
            if step % 3001 == 1 and step != 1:
                time.sleep(45)
            # move data to GPU
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            # print("inputs.shape:", inputs.shape)
            # zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            # print("about to get model output")
            outputs = model(inputs)
            # print("done getting model output")
            # print("outputs.shape:", outputs.shape, "targets.shape:", targets.shape)
            loss = criterion(outputs, targets)
            # Backward and optimize
            # print("about to optimize")
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        # Get train loss and test loss
        train_loss = np.mean(train_loss)  # a little misleading

        model.eval()
        test_loss = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss

        if test_loss < best_test_loss:
            torch.save(model, os.path.join(data_path, 'best_val_model_pytorch'))
            best_test_loss = test_loss
            best_test_epoch = it
            print('model saved')

        dt = datetime.now() - t0
        print(f'Epoch {it + 1}/{epochs}, Train Loss: {train_loss:.4f}, \
          Validation Loss: {test_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch}')

    return train_losses, test_losses


if __name__ == "__main__":
    args = parse_args()

    data_path = args.data_path
    train_df_path = os.path.join(args.data_path, args.train_file)
    valid_df_path = os.path.join(args.data_path, args.valid_file)
    test_df_path = os.path.join(args.data_path, args.test_file)

    dec_val = np.genfromtxt(valid_df_path, delimiter=',', skip_header=1)
    dec_test = np.genfromtxt(test_df_path, delimiter=',', skip_header=1)
    dec_train = np.genfromtxt(train_df_path, delimiter=',', skip_header=1)

    dataset_train = Dataset(data=dec_train, num_classes=3, T=100)
    dataset_val = Dataset(data=dec_val, num_classes=3, T=100)
    dataset_test = Dataset(data=dec_test, num_classes=3, T=100)

    batch_size = 64
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device {device}")

    model = deeplob(y_len=dataset_train.num_classes)
    # model = nn.DataParallel(model)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_losses, val_losses = batch_gd(model, criterion, optimizer,
                                        train_loader, val_loader, epochs=100)

    # test
    model = torch.load(os.path.join(data_path, 'best_val_model_pytorch'))

    model.eval()

    predictions = []
    true_labels = []
    total_samples_from_loader = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            total_samples_from_loader += len(labels)
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.int64)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    print(f"len loader {total_samples_from_loader}")
    print(f"len predictions {len(predictions)}")

    cm = confusion_matrix(true_labels, predictions)
    cm_percentage = cm / cm.sum(axis=0, keepdims=True) * 100

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax[0])
    ax[0].set_title('Confusion Matrix (Counts)')
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', ax=ax[1])
    ax[1].set_title('Confusion Matrix (Percentage)')
    plt.savefig(os.path.join(data_path, "confusion.png"))
    # plt.show()

    accuracy = accuracy_score(true_labels, predictions) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    precision *= 100
    recall *= 100
    f1 *= 100

    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Precision: {precision:.2f}%')
    print(f'Recall: {recall:.2f}%')
    print(f'F1 Score: {f1:.2f}%')

    df_predictions = pd.DataFrame({'PredictedLabels': predictions})
    predictions_path = os.path.join(data_path, 'predictions.csv')  # 預測結果保存的路徑
    df_predictions.to_csv(predictions_path, index=False)
    df_trues = pd.DataFrame({'TruesLabels': true_labels})
    trues_path = os.path.join(data_path, 'true_labels.csv')  # 預測結果保存的路徑
    df_trues.to_csv(trues_path, index=False)

