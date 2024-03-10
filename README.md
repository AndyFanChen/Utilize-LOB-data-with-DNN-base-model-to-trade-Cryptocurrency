# Utilize LOB data with DNN base model to trade Cryptocurrency

## Task Description
I am trying to use deep neural network model to find the information in LOB(Limit order book) to predict the trend of cryptocurrency price. I try to utilize two model in two paper, which are DeepLOB by Zhang, Zohren and Roberts([paper link](https://arxiv.org/abs/1808.03668)) and Axial-LOB by Kisiel and Gorse([paper link](https://arxiv.org/abs/2212.01807)). In DeepLOB, they use CNN to construct the model. In Axial-LOB they use transformer to construct the model.

## Result
I make the profit 3% more then purely hold strategy during about 2 days. Plot of profit of my strategy:
![strategy_return](https://github.com/AndyFanChen/Utilize-LOB-data-with-DNN-base-model-to-trade-Cryptocurrency/assets/99866172/4276f5b1-00e2-484c-ab6c-f554ac778c81)



## Data
I use the data from kaggle, link of the data is: [link](https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data?select=BTC_1sec.csv), there are three kinds of cryptocurrency LOB data.

## Execute Step
Use `btc_data_process.py` to modify the data to be the format which can be input of the model, use `deeplob.py` and `axial_lob.py` can use the code modify from [DeepLOB Github](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books) and [Axial-LOB Github](https://github.com/LeonardoBerti00/Axial-LOB-High-Frequency-Trading-with-Axial-Attention), to train and save the model. After training, can use `deep_lob_inference.py` and `axial_lob_inference.py`  to use the result then see backtest result by the trading strategy using the model trained.


