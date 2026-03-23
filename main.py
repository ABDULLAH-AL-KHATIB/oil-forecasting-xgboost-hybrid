"""
The Accuracy-Direction Trade-Off in Energy Markets
Code repository for the paper: "XGBoost Outperforms Deep Learning in Oil Price Forecasting"
Models included: ARIMA, Random Forest, XGBoost, LightGBM, LSTM, CNN-LSTM, ARIMA-XGBoost Hybrid.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os

import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout
import shap
import scipy.stats as stats

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

class DataProcessor:
    def __init__(self, file_path=None):
        self.file_path = file_path

    def load_or_fetch_data(self):
        """Loads custom dataset if path provided, else fetches from Yahoo Finance."""
        if self.file_path and os.path.exists(self.file_path):
            print(f"Loading data from {self.file_path}...")
            df = pd.read_csv(self.file_path, parse_dates=True, index_col='Date')
        else:
            print("Fetching data from Yahoo Finance API...")
            tickers = {'Brent': 'BZ=F', 'WTI': 'CL=F', 'NatGas': 'NG=F', 
                       'DXY': 'DX-Y.NYB', 'VIX': '^VIX', 'Yield10Y': '^TNX'}
            df = pd.DataFrame()
            start_date = '2016-05-24'
            end_date = datetime.today().strftime('%Y-%m-%d')
            for name, ticker in tickers.items():
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                df[name] = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
            df.fillna(method='ffill', inplace=True)
            df.dropna(inplace=True)
        return df

    def engineer_features(self, df, target='Brent'):
        print("Engineering Macroeconomic and Technical Features...")
        data = df.copy()
        data['Log_Return'] = np.log(data[target] / data[target].shift(1))
        data['Vol_7d'] = data['Log_Return'].rolling(window=7).std()
        data['Vol_30d'] = data['Log_Return'].rolling(window=30).std()
        data['MA_10'] = data[target].rolling(window=10).mean()
        data['MA_50'] = data[target].rolling(window=50).mean()
        data['RSI'] = ta.momentum.RSIIndicator(data[target], window=14).rsi()
        data['MACD'] = ta.trend.MACD(data[target]).macd()
        
        for i in [1, 2, 3, 5]:
            data[f'Lag_{i}'] = data[target].shift(i)
        data.dropna(inplace=True)
        return data

class HybridForecaster:
    def evaluate(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        mape = mean_absolute_percentage_error(actual, pred) * 100
        da = np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(pred))) * 100
        return rmse, mae, mape, da

    def dm_test(self, actual, pred1, pred2):
        """Diebold-Mariano Statistical Test."""
        e1, e2 = actual - pred1, actual - pred2
        d = e1**2 - e2**2
        var_d = np.var(d, ddof=1)
        if var_d == 0: return 0, 1
        stat = np.mean(d) / np.sqrt(var_d / len(d))
        pval = 2 * (1 - stats.norm.cdf(abs(stat)))
        return stat, pval

def run_pipeline():
    processor = DataProcessor() # Add your CSV path here e.g., DataProcessor('data.csv')
    df_raw = processor.load_or_fetch_data()
    df = processor.engineer_features(df_raw)
    
    target = 'Brent'
    features = [col for col in df.columns if col not in [target, 'Log_Return']]
    test_size = 150 # Out-of-sample period (Structural break)
    look_back = 10
    
    train_df, test_df = df.iloc[:-test_size], df.iloc[-test_size:]
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    actuals = y_test.values

    # Scaling for Deep Learning
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

    # DL Sequence prep
    X_full_scaled = scaler_X.transform(df[features])
    y_full_scaled = scaler_y.transform(df[target].values.reshape(-1, 1))
    Xs, ys = [], []
    for i in range(len(X_full_scaled) - look_back):
        Xs.append(X_full_scaled[i:(i + look_back)])
        ys.append(y_full_scaled[i + look_back])
    X_dl, y_dl = np.array(Xs), np.array(ys)
    X_train_dl, y_train_dl = X_dl[:-test_size], y_dl[:-test_size]
    X_test_dl = X_dl[-test_size:]

    print("Training Models...")
    preds = {}

    # 1. Statistical: ARIMA
    history = list(y_train.values)
    arima_p = []
    for t in range(test_size):
        model = ARIMA(history, order=(5,1,0)).fit()
        arima_p.append(model.forecast()[0])
        history.append(actuals[t])
    preds['ARIMA'] = np.array(arima_p)

    # 2. Machine Learning: RF, XGB, LGBM
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42).fit(X_train, y_train)
    lgbm = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1).fit(X_train, y_train)
    
    preds['RandomForest'] = rf.predict(X_test)
    preds['XGBoost'] = xgb.predict(X_test)
    preds['LightGBM'] = lgbm.predict(X_test)

    # 3. Deep Learning: LSTM & CNN-LSTM
    lstm = Sequential([LSTM(50, activation='relu', input_shape=(look_back, len(features))), Dropout(0.2), Dense(1)])
    lstm.compile(optimizer='adam', loss='mse')
    lstm.fit(X_train_dl, y_train_dl, epochs=30, batch_size=32, verbose=0)
    preds['LSTM'] = scaler_y.inverse_transform(lstm.predict(X_test_dl, verbose=0)).flatten()

    cnn_lstm = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(look_back, len(features))),
        MaxPooling1D(pool_size=2), LSTM(50, activation='relu'), Dropout(0.2), Dense(1)])
    cnn_lstm.compile(optimizer='adam', loss='mse')
    cnn_lstm.fit(X_train_dl, y_train_dl, epochs=30, batch_size=32, verbose=0)
    preds['CNN-LSTM'] = scaler_y.inverse_transform(cnn_lstm.predict(X_test_dl, verbose=0)).flatten()

    # 4. Hybrid: ARIMA-XGBoost
    arima_train = ARIMA(y_train.values, order=(5,1,0)).fit()
    residuals = y_train.values - arima_train.fittedvalues
    xgb_resid = XGBRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_train, residuals)
    preds['ARIMA-XGBoost'] = preds['ARIMA'] + xgb_resid.predict(X_test)

    # Results Evaluation
    forecaster = HybridForecaster()
    print("\n--- Model Evaluation ---")
    print(f"{'Model':<15} | {'RMSE':<6} | {'MAE':<6} | {'MAPE':<6} | {'DA (%)':<6}")
    for name, p in preds.items():
        rmse, mae, mape, da = forecaster.evaluate(actuals, p)
        print(f"{name:<15} | {rmse:<6.2f} | {mae:<6.2f} | {mape:<6.2f} | {da:<6.2f}")

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(test_df.index, actuals, label='Actual Brent Price', color='black', linewidth=2)
    plt.plot(test_df.index, preds['XGBoost'], label='XGBoost', linestyle='dotted')
    plt.plot(test_df.index, preds['CNN-LSTM'], label='CNN-LSTM', alpha=0.7)
    plt.plot(test_df.index, preds['ARIMA-XGBoost'], label='ARIMA-XGBoost Hybrid', color='red')
    plt.title('Out-of-Sample Forecasting During Structural Break')
    plt.legend()
    plt.show()

    # SHAP
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)

if __name__ == "__main__":
    run_pipeline()