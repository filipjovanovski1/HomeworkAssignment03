import os
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import datetime
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import ta


def analyze_technical_data(stock_symbol: str) -> pd.DataFrame:
    url = f'https://www.mse.mk/mk/akcija/{stock_symbol}/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    table = soup.find_all('table')[1]
    headers = [header.text for header in table.find_all('th') if header.text]
    
    rows = []
    for row in table.find_all('tr'):
        columns = row.find_all('td')
        row_data = [col.text.strip() for col in columns]
        if row_data:
            rows.append(row_data)

    stock_data = pd.DataFrame(rows, columns=headers)
    stock_data.rename(columns=lambda x: x.strip(), inplace=True)

    stock_data['DATE'] = pd.to_datetime(stock_data['DATE'], format='%d.%m.%Y')
    stock_data['PRICE'] = stock_data['PRICE OF LAST TRANSACTION IN mkd']
    
    for column in ['PRICE', 'MIN', 'MAX']:
        stock_data[column] = stock_data[column].replace('', pd.NA)
        stock_data[column] = stock_data[column].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
    
    stock_data.sort_values('DATE', inplace=True)
    stock_data['PRICE'].fillna(method='ffill', inplace=True)
    stock_data['PRICE'].fillna(method='bfill', inplace=True)

    stock_data['RSI'] = ta.momentum.RSIIndicator(stock_data['PRICE']).rsi()
    stock_data['CCI'] = ta.trend.CCIIndicator(stock_data['HIGH'], stock_data['LOW'], stock_data['PRICE']).cci()
    stock_data['%K'] = ta.momentum.StochasticOscillator(stock_data['HIGH'], stock_data['LOW'], stock_data['PRICE']).stoch()

    stock_data['Signal'] = stock_data['RSI'].apply(lambda x: 'Buy' if x < 30 else 'Sell' if x > 70 else 'Hold')

    return stock_data


def load_and_prepare_data(stock_data: pd.DataFrame) -> tuple:
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(stock_data[['PRICE']])
    
    sequence_length = 60
    x_train, y_train = [], []
    
    for i in range(sequence_length, len(scaled_data)):
        x_train.append(scaled_data[i - sequence_length:i])
        y_train.append(scaled_data[i])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    return x_train, y_train, scaler


def create_lstm_model(input_shape: tuple) -> Sequential:
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


def plot_predictions(y_test: np.ndarray, predictions: np.ndarray, scaler: MinMaxScaler, stock_symbol: str) -> None:
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    predictions_unscaled = scaler.inverse_transform(predictions)

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_unscaled, label='Actual Prices')
    plt.plot(predictions_unscaled, label='Predicted Prices')
    plt.title(f'Stock Price Prediction for {stock_symbol}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(f"data/lstm_plot_{stock_symbol}.png")
    plt.close()
