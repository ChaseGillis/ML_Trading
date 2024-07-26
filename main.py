import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier

# Data Collection
def download_data(ticker):
    return yf.download(ticker, start='2010-01-01', end='2023-01-01')

# Feature Engineering
def create_features(data):
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data = data.dropna()
    return data

# Model Training and Backtesting
def train_and_backtest(data):
    features = data[['MA50', 'MA200', 'RSI']]
    target = data['Close'].shift(-1) > data['Close']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    def backtest(data, model, X_test):
        predictions = model.predict(X_test)
        data = data[-len(X_test):].copy()
        data['Predictions'] = predictions
        data['Strategy'] = np.where(data['Predictions'] > 0.5, data['Close'].shift(-1) - data['Close'], 0)
        data['Market'] = data['Close'].shift(-1) - data['Close']
        return data

    backtest_results = backtest(data, model, X_test)

    def calculate_metrics(backtest_results):
        daily_returns = backtest_results['Strategy']
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        average_roi = (backtest_results['Strategy'].sum() / len(backtest_results)) * 252 / data['Close'][0] * 100
        return sharpe_ratio, average_roi

    sharpe_ratio, average_roi = calculate_metrics(backtest_results)
    print(f'Sharpe Ratio: {sharpe_ratio}')
    print(f'Average ROI: {average_roi}%')
    return sharpe_ratio, average_roi

# Evaluate across multiple stocks
stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
for ticker in stocks:
    print(f'\nEvaluating {ticker}')
    data = download_data(ticker)
    data = create_features(data)
    sharpe_ratio, average_roi = train_and_backtest(data)
    print(f'{ticker} - Sharpe Ratio: {sharpe_ratio}, Average ROI: {average_roi}%')