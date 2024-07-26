import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Data Collection
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')

# Feature Engineering
def create_features(data):
    # Calculate moving averages
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    # Calculate Relative Strength Index (RSI)
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Drop NaN values
    data = data.dropna()
    
    return data

data = create_features(data)

# Model Training
features = data[['MA50', 'MA200', 'RSI']]
target = data['Close'].shift(-1) > data['Close']  # Predict if the next day's closing price will be higher

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Backtesting
def backtest(data, model, X_test):
    predictions = model.predict(X_test)
    data = data[-len(X_test):].copy()
    data['Predictions'] = predictions
    data['Strategy'] = np.where(data['Predictions'] > 0.5, data['Close'].shift(-1) - data['Close'], 0)
    data['Market'] = data['Close'].shift(-1) - data['Close']
    
    return data

backtest_results = backtest(data, model, X_test)

# Metrics Calculation
def calculate_metrics(backtest_results):
    daily_returns = backtest_results['Strategy']
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    average_roi = (backtest_results['Strategy'].sum() / len(backtest_results)) * 252 / data['Close'][0] * 100
    
    return sharpe_ratio, average_roi

sharpe_ratio, average_roi = calculate_metrics(backtest_results)
print(f'Sharpe Ratio: {sharpe_ratio}')
print(f'Average ROI: {average_roi}%')
