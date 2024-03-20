import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyfolio as pf
import backtrader as bt


data_path = r'./data/BTC_1sec_test.csv'
data = pd.read_csv(data_path)

for col in data.columns:
    if 'bids_distance_' in col or 'asks_distance_' in col:
        data[col] = data['midpoint'] * (1 + data[col])

T = 100

data['change'] = data['midpoint'].rolling(window=T).mean().shift(-T) - data['midpoint'].rolling(window=T).mean()

data.dropna(inplace=True)
data = data.iloc[T-1:]

label_path = './data/predictions_best.csv'  # 標籤資料的路徑
labels = pd.read_csv(label_path, header=0)  # 假設標籤CSV沒有標題行

print(f"len(data) {len(data)}")
print(f"len(labels) {len(labels)}")
data['label'] = labels.values

def cal_profit_2(data):

    signal_set(data)
    data['strategy_returns'] = np.where(
        data['signal'].shift(1) == 1,  # Long
        data['midpoint'] / data['midpoint'].shift(1) - 1,
        np.where(
            data['signal'].shift(1) == -1,  # Short
            data['midpoint'].shift(1) / data['midpoint']  - 1,
            0
    )
    returns = data['strategy_returns'].dropna()

    return returns

def signal_set(data):
    signals = np.zeros(len(data))
    signals = np.where(data['label'].values == 2, 1, signals)
    signals = np.where(data['label'].values == 0, -1, signals)
    signals = np.where(data['label'].values == 1, np.nan, signals)

    for i in range(1, len(signals)):
        if np.isnan(signals[i]):
            signals[i] = signals[i - 1]

    data['signal'] = signals

date = data['system_time']
buy_hold_data = pd.DataFrame({'midpoint': data['midpoint'], 'label': [0] * len(data)})
returns = cal_profit_2(data)
buy_hold_returns = cal_profit_2(buy_hold_data)

system_time = pd.to_datetime(date)

df = pd.DataFrame({'system_time': system_time, 'strategy_returns': returns}).set_index('system_time')

df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod() - 1

str_return = df['cumulative_returns'].iloc[-1]
plt.figure(figsize=(14, 7))

plt.plot(df.index, df['cumulative_returns'], label='Strategy', color='blue')
plt.title('Strategy Cumulative Returns')
plt.legend()

df = pd.DataFrame({'system_time': system_time, 'strategy_returns': buy_hold_returns}).set_index('system_time')

df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod() - 1
hold_return = df['cumulative_returns'].iloc[-1]
print(f"dif = {hold_return - str_return}")

plt.plot(df.index, df['cumulative_returns'], label='Holding Short Position', color='red')
plt.title('Holding Short Position V.S. Strategy Cumulative Returns')
plt.legend()

plt.tight_layout()
plt.savefig('strategy_return.png', dpi=300)
plt.show()
