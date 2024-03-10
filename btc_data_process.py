import pandas as pd
from sklearn.model_selection import train_test_split

# read the data
file_path = r'../data/BTC_1sec.csv'  # 更改为你的CSV文件路径
df = pd.read_csv(file_path)

# bids_limit_notional
columns_to_keep = ['system_time'] + \
                  ['midpoint'] + \
                  [f'bids_distance_{i}' for i in range(10)] + \
                  [f'asks_distance_{i}' for i in range(10)] + \
                  [f'bids_limit_notional_{i}' for i in range(10)] + \
                  [f'asks_limit_notional_{i}' for i in range(10)]
df_filtered = df[columns_to_keep]

# split train, valid, test set by 70:10:20
temp_train_df, test_df = train_test_split(df_filtered, test_size=0.2, random_state=42)

train_df, valid_df = train_test_split(temp_train_df, test_size=(1/8), random_state=42)

# 4. 将三组新数据分别保存到新的CSV文件
train_df.to_csv(r'../data/BTC_1sec_train_0310.csv', index=False)
valid_df.to_csv(r'../data/BTC_1sec_valid_0310.csv', index=False)
test_df.to_csv(r'../data/BTC_1sec_test_0310.csv', index=False)

# # Or use date to split data

# file_path = r'./data/ETH_1sec.csv'
# data = pd.read_csv(file_path)
# print(f"len(data) {len(data)}")
#
# # Select the required columns
# columns_to_keep = ['system_time'] + \
#                   ['midpoint'] + \
#                   [column_name for i in range(10) for column_name in (
#                       f'bids_distance_{i}', f'bids_notional_{i}',
#                       f'asks_distance_{i}', f'asks_notional_{i}')]
#
# data_filtered = data[columns_to_keep]
#
# # Convert system_time to datetime
# data_filtered['system_time'] = pd.to_datetime(data_filtered['system_time'])
#
# # # Filter data into train, validation, and test sets
# train_set = data_filtered[(data_filtered['system_time'] < '2021-04-16')]
# valid_set = data_filtered[(data_filtered['system_time'] >= '2021-04-16') & (data_filtered['system_time'] < '2021-04-18')]
# test_set = data_filtered[data_filtered['system_time'] >= '2021-04-18']
# # print(f"len(train) {len(train_set)}")
# # print(f"len(valid) {len(valid_set)}")
# # print(f"len(test) {len(test_set)}")
# # print(f"len total = {len(train_set) + len(valid_set) + len(test_set)}")
#
#
# # Save the new datasets to CSV files
# train_set.to_csv(r'./data/ETH_1sec_train.csv', index=False)
# valid_set.to_csv(r'./data/ETH_1sec_valid.csv', index=False)
# test_set.to_csv(r'./data/ETH_1sec_test.csv', index=False)
# print(f"pre save over!")

# =====================================================================================
# read the data again
train_df_path = r'../data/BTC_1sec_train_0310.csv'
valid_df_path = r'../data/BTC_1sec_valid_0310.csv'
test_df_path = r'../data/BTC_1sec_test_0310.csv'

# train_df = pd.read_csv(train_df_path)
# valid_df = pd.read_csv(valid_df_path)
test_df = pd.read_csv(test_df_path)

def process_data(df):
    for col in df.columns:
        if 'bids_distance_' in col or 'asks_distance_' in col:
            df[col] = df['midpoint'] * (1 + df[col])

    t = 100 # size of the window
    df['change'] = df['midpoint'].rolling(window=t).mean().shift(-t) - df['midpoint'].rolling(window=t).mean()

    df.drop(columns=['midpoint'], inplace=True)
    df.dropna(inplace=True)

    return df

train_df = process_data(train_df)
valid_df = process_data(valid_df)
test_df = process_data(test_df)


def standardize_data(train_df, valid_df, test_df):

    features_train = train_df.iloc[:, 1:]
    features_valid = valid_df.iloc[:, 1:]
    features_test = test_df.iloc[:, 1:]

    mean = features_train.mean()
    std = features_train.std()

    train_df_standardized = (features_train - mean) / std
    valid_df_standardized = (features_valid - mean) / std
    test_df_standardized = (features_test - mean) / std

    return train_df_standardized, valid_df_standardized, test_df_standardized


def label_creat(df):

    a = 0.3 # threshold to set the label
    df['label'] = 1
    df.loc[df['change'] > a, 'label'] = 2
    df.loc[df['change'] < -a, 'label'] = 0

    df.drop(columns=['change'], inplace=True)

    label_counts = df['label'].value_counts()
    total_counts = len(df)
    label_proportions = label_counts / total_counts
    print("ratio of 0 in label:", label_proportions[0])
    print("ratio of 1 in label:", label_proportions[1])
    print("ratio of 2 in label:", label_proportions[2])

    return df

train_df, valid_df, test_df = standardize_data(train_df, valid_df,
                                               test_df)
train_df = label_creat(train_df)
valid_df = label_creat(valid_df)
test_df = label_creat(test_df)
print(len(test_df))

#
# save data
train_df.to_csv(r'./data/BTC_1sec_processed_train.csv', index=False)
valid_df.to_csv(r'./data/BTC_1sec_processed_valid.csv', index=False)
test_df.to_csv(r'./data/BTC_1sec_processed_test.csv', index=False)