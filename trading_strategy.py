import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyfolio as pf
import backtrader as bt

# 1. 從 CSV 檔讀取資料a
data_path = r'./data/BTC_1sec_test_0310.csv'  # 資料a的路徑
data = pd.read_csv(data_path)

# 2. 丟棄 `system_time` 列
# data.drop('system_time', axis=1, inplace=True)
for col in data.columns:
    if 'bids_distance_' in col or 'asks_distance_' in col:
        data[col] = data['midpoint'] * (1 + data[col])
# 假定 T 的值
T = 100  # 可以根據需求調整

# 3. 丟棄前 T-1 筆和最後 T-1 筆資料

data['change'] = data['midpoint'].rolling(window=T).mean().shift(-T) - data['midpoint'].rolling(window=T).mean()

# 步骤4：删除midpoint列
# data.drop(columns=['midpoint'], inplace=True)
data.dropna(inplace=True)
data = data.iloc[T-1:]


# data2_path = r'./data/BTC_1sec_processed_test_2.csv'  # 資料a的路徑
# data_2 = pd.read_csv(data2_path)
# print(f"len(data2) {len(data_2)}")
# slice_data2 = data_2.iloc[T-1:]
#
# 4. 從另一個 CSV 檔讀取標籤資料
label_path = './data/predictions_best.csv'  # 標籤資料的路徑
labels = pd.read_csv(label_path, header=0)  # 假設標籤CSV沒有標題行

print(f"len(data) {len(data)}")
print(f"len(labels) {len(labels)}")
# 確保標籤筆數與資料筆數相同
# labels = labels.iloc[T-1:-T+1]

# 將標籤合併到資料中
data['label'] = labels.values

# =================================================================
# import torch
# import os
# import numpy as np
# import pandas as pd
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# true_labels = slice_data2['label']
# predictions = data['label']
#
# cm = confusion_matrix(true_labels, predictions)
# cm_percentage = cm / cm.sum(axis=0, keepdims=True) * 100
#
# # 繪製並顯示混淆矩陣
# fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# sns.heatmap(cm, annot=True, fmt='d', ax=ax[0])
# ax[0].set_title('Confusion Matrix (Counts)')
# sns.heatmap(cm_percentage, annot=True, fmt='.2f', ax=ax[1])
# ax[1].set_title('Confusion Matrix (Percentage)')
# plt.show()
#
# # 計算評估指標
# accuracy = accuracy_score(true_labels, predictions) * 100
# precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
# precision *= 100
# recall *= 100
# f1 *= 100
#
# print(f'Accuracy: {accuracy:.2f}%')
# print(f'Precision: {precision:.2f}%')
# print(f'Recall: {recall:.2f}%')
# print(f'F1 Score: {f1:.2f}%')

# =======================================================================================
# class TestStrategy(bt.Strategy):
#     def log(self, txt, dt=None):
#         dt = dt or self.datas[0].datetime.date(0)
#         print('%s, %s' % (dt.isoformat(), txt))
#
#     def __init__(self):
#         # 假设数据中包含label列
#         self.label = self.datas[0].label
#
#     def next(self):
#         # 检查当前的label值，并执行相应的交易逻辑
#         if self.label[0] == 1:
#             # 如果当前未持仓且label为1，则做多
#             if not self.position:
#                 self.log('BUY CREATE, %.2f' % self.dataclose[0])
#                 self.buy()
#         elif self.label[0] == -1:
#             # 如果当前持仓且label为-1，则做空
#             if not self.position:
#                 self.log('SELL CREATE, %.2f' % self.dataclose[0])
#                 self.sell()
#         elif self.label[0] == 0:
#             # 如果label为0，则不进行交易
#             pass
#
# class CustomDataFeed(bt.feeds.PandasData):
#     params = (
#         ('datetime', 'system_time'),
#         ('label', 'label'),  # 确保label列被加载
#     )

# data['system_time'] = pd.to_datetime(data['system_time'])
# cerebro = bt.Cerebro()
# datafeed = CustomDataFeed(dataname=data)
# cerebro.adddata(datafeed)
# cerebro.addstrategy(TestStrategy)
# cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")
# results = cerebro.run()
# time_return_analyzer = results[0].analyzers.getbyname('time_return')
# returns = time_return_analyzer.get_analysis()




def cal_profit_2(data):
    # 生成交易信号：高于10日移动平均线则Long（1），否则Short（-1）
    signal_set(data)

    # data['strategy_returns'] = np.where(
    #     data['signal'].shift(1) == 1,  # Long: 买入价格为上一天的卖价，卖出价格为当天的买价
    #     data['asks_distance_0'] / data['bids_distance_0'].shift(1) - 1,
    #     np.where(
    #         data['signal'].shift(1) == -1,  # Short: 买入价格为当天的卖价，卖出价格为上一天的买价
    #         data['asks_distance_0'] /  data['bids_distance_0'].shift(1) - 1,
    #         0  # 不进行交易的情况
    #     )
    # )

    data['strategy_returns'] = np.where(
        data['signal'].shift(1) == 1,  # Long: 买入价格为上一天的卖价，卖出价格为当天的买价
        data['midpoint'] / data['midpoint'].shift(1) - 1,
        np.where(
            data['signal'].shift(1) == -1,  # Short: 买入价格为当天的卖价，卖出价格为上一天的买价
            data['midpoint'].shift(1) / data['midpoint']  - 1,
            0  # 不进行交易的情况
        )
    )

    # 使用pyfolio分析策略表现
    returns = data['strategy_returns'].dropna()
    #
    # start_date = "1990-04-18"

    # 创建日期范围
    # periods参数设置为收益率序列的长度
    # date_range = pd.date_range(start=start_date, periods=len(returns), freq='D')

    # 创建包含日期和收益率的DataFrame
    # df = pd.DataFrame({'Date': date_range, 'Strategy Returns': returns})
    # df = df.set_index('Date')
    # df.dropna(inplace=True)
    # result = pf.create_full_tear_sheet(df, benchmark_rets=None)

    return returns

def signal_set(data):
    # 初始化signal数组
    signals = np.zeros(len(data))

    # 处理label等于2和0的情况
    signals = np.where(data['label'].values == 2, 1, signals)
    signals = np.where(data['label'].values == 0, -1, signals)

    # signals = np.where(data['label'].values == 2, 1, signals)
    # signals = np.where(data['label'].values == 0, -1, signals)

    # 处理label等于1的情况
    # 先标记为NaN
    signals = np.where(data['label'].values == 1, np.nan, signals)

    # 填充NaN为前一个非NaN的值
    for i in range(1, len(signals)):
        if np.isnan(signals[i]):
            signals[i] = signals[i - 1]

    # 更新DataFrame
    data['signal'] = signals

date = data['system_time']
buy_hold_data = pd.DataFrame({'midpoint': data['midpoint'], 'label': [0] * len(data)})
returns = cal_profit_2(data)
buy_hold_returns = cal_profit_2(buy_hold_data)


# 将时间字符串转换为datetime对象
system_time = pd.to_datetime(date)

# 创建DataFrame
df = pd.DataFrame({'system_time': system_time, 'strategy_returns': returns}).set_index('system_time')

# 计算累积收益率
df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod() - 1

str_return = df['cumulative_returns'].iloc[-1]
# 绘制收益率时间序列
plt.figure(figsize=(14, 7))

# plt.subplot(2, 1, 1)
# plt.plot(df.index, df['strategy_returns'], label='Strategy Returns', color='blue')
# plt.title('Strategy Returns Over Time')
# plt.legend()

# 绘制累积收益率时间序列
# plt.subplot(2, 1, 1)
plt.plot(df.index, df['cumulative_returns'], label='Strategy', color='blue')
plt.title('Strategy Cumulative Returns')
plt.legend()

# plt.tight_layout()  # 调整子图间距
# plt.show()


df = pd.DataFrame({'system_time': system_time, 'strategy_returns': buy_hold_returns}).set_index('system_time')

# 计算累积收益率
df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod() - 1
hold_return = df['cumulative_returns'].iloc[-1]
print(f"dif = {hold_return - str_return}")
# # 绘制收益率时间序列
# plt.figure(figsize=(14, 7))
#
# plt.subplot(2, 1, 1)
# plt.plot(df.index, df['strategy_returns'], label='Strategy Returns', color='blue')
# plt.title('Strategy Returns Over Time')
# plt.legend()

# 绘制累积收益率时间序列
# plt.subplot(2, 1, 2)
plt.plot(df.index, df['cumulative_returns'], label='Holding Short Position', color='red')
plt.title('Holding Short Position V.S. Strategy Cumulative Returns')
plt.legend()

plt.tight_layout()  # 调整子图间距
plt.savefig('strategy_return.png', dpi=300)
plt.show()

# 計算報酬率
# profits, profits_idx, profits_len, profit_rates, cum_profits, cum_profit_rates = calculate_profit_rate(data)
# # total_rate = (sum(profits) / sum(total_in)) - 1
# # print(f"my_profit_rate {cum_profit_rates[-1]}")
# # 計算 Buy and Hold 累積報酬率
# buy_price = data['asks_distance_0'].iloc[0]
# sell_price = data['asks_distance_0'].iloc[-1]
# buy_and_hold_profit_rate = ((sell_price - buy_price) / buy_price)
# print(f"buy_and_hold_profit_rate {buy_and_hold_profit_rate} profit {sell_price - buy_price}")
#
# fig, ax = plt.subplots()
# ax.plot(profits_idx, cum_profits)
# plt.show()
# fig, ax = plt.subplots()
# ax.plot(profits_idx, profits)
# plt.show()
#
# # ori_profits, ori_profit_rates = profits_ori(data['midpoint'])
# # ori_cum_profit_rates = cumulative_profits_cal(ori_profit_rates)
# fig2, ax2 = plt.subplots()
# ax2.set_ylim(-0.5, 0.5)
# ax2.plot(data['midpoint'].pct_change())
#
# # ax.plot(ori_cum_profit_rates)
# plt.show()
