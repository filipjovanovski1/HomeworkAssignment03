from json import JSONEncoder

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import os
import time
from flask import Flask, jsonify

from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from ta.trend import CCIIndicator


start_time = time.time()

app = Flask(__name__)

def format_date(date_str):
    date_obj = datetime.strptime(date_str, '%m/%d/%Y')
    return date_obj.strftime('%d.%m.%Y')

def fetch_issuer_codes():
    url = "https://www.mse.mk/en/stats/symbolhistory/kmb"
    with requests.Session() as session:
        response = session.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    dropdown = soup.find("select", id="Code")

    if not dropdown:
        print("No issuer codes found.")
        return []

    return [
        option.text.strip()
        for option in dropdown.find_all("option")
        if not any(char.isdigit() for char in option.text)
    ]

def check_last_date(issuer_code):
    file_path = f"csv/{issuer_code}.csv"
    try:
        df = pd.read_csv(file_path)
        return pd.to_datetime(df['Date'], format='%d.%m.%Y').max()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None

def fetch_data(issuer_code, start_date, end_date):
    url = (
        f"https://www.mse.mk/en/stats/symbolhistory/{issuer_code}"
        f"?FromDate={start_date.strftime('%m/%d/%Y')}"
        f"&ToDate={end_date.strftime('%m/%d/%Y')}"
    )
    with requests.Session() as session:
        response = session.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    tbody = soup.select_one('tbody')

    if not tbody:
        print(f"No data found for {issuer_code} from {start_date} to {end_date}.")
        return []

    return [
        [cell.get_text(strip=True) for cell in row.find_all('td')]
        for row in tbody.find_all('tr')
    ]


def safe_convert_to_float(data):
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)  # Ensure correct date parsing
    data = data.replace('', pd.NA)  # Replace empty strings with NaN

    for column in data.columns:
        if data[column].dtype == "O":  # Only process object (string) columns
            data[column] = (
                data[column]
                .str.replace(',', '', regex=False)  # Remove thousand separators
                .astype(float)
            )

    data["Min"] = data["Min"].fillna(data["Close"])
    data["Max"] = data["Max"].fillna(data["Close"])

    return data


def save_to_csv(issuer_code, data):
    columns = ['Date', 'Close', 'Max', 'Min', 'Avg. Price', '%chg.', 'Volume', 'Turnover in BEST', 'Total turnover']
    df = pd.DataFrame(data, columns=columns)
    df['Date'] = df['Date'].apply(format_date)

    file_path = f"csv/{issuer_code}.csv"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
    print(f"Data saved to {file_path}")

def update_data(issuer_code, last_date):
    current_date = datetime.now()
    start_date = last_date + timedelta(days=1) if last_date else current_date - timedelta(days=3650)

    while start_date <= current_date:
        end_date = min(datetime(start_date.year, 12, 31), current_date) if (current_date.year - start_date.year) >= 1 else current_date
        data = fetch_data(issuer_code, start_date, end_date)
        if data:
            save_to_csv(issuer_code, data)
        start_date = end_date + timedelta(days=1)
        if end_date == current_date:
            break


def calculate_technical_indicators(df):
    df = safe_convert_to_float(df)

    indicators = {}

    indicators['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    indicators['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    indicators['EMA_20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    indicators['EMA_50'] = EMAIndicator(df['Close'], window=50).ema_indicator()
    indicators['BB_Mid'] = BollingerBands(df['Close'], window=20).bollinger_mavg()

    indicators['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    indicators['OBV'] = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    indicators['Momentum'] = df['Close'].diff(periods=10)

    # **Add CCI calculation** (this is missing in your original code)
    indicators['CCI'] = CCIIndicator(high=df['Max'], low=df['Min'], close=df['Close'], window=20).cci()

    for name, values in indicators.items():
        df[name] = values

    # Signals based on indicators
    df['Signal'] = df['RSI'].apply(lambda x: 'Buy' if x < 30 else 'Sell' if x > 70 else 'Hold')
    df['CCI_Signal'] = df['CCI'].apply(lambda x: 'Buy' if x < -100 else 'Sell' if x > 100 else 'Hold')

    print(f"Indicators columns: {df.columns}")

    return df


@app.route('/api/get/transactions/<string:issuer_code>', methods=["GET"])
def get_data(issuer_code):
    file_path = f"csv/{issuer_code}.csv"
    df = pd.read_csv(file_path)
    df = calculate_technical_indicators(df)

    json_str = df.to_json(orient='records')
    return app.response_class(json_str, mimetype='application/json')

@app.route('/api/get/codes', methods=["GET"])
def get_codes():
    codes = fetch_issuer_codes()
    return jsonify(codes)

def main():
    issuer_codes = fetch_issuer_codes()
    if not os.path.exists("csv"):
        for code in issuer_codes:
            last_date = check_last_date(code)
            update_data(code, last_date)
        print("Data retrieval and storage complete.")
        print(f"Execution Time: {(time.time() - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    main()
    app.run(host='0.0.0.0', port = 5000, debug = True)
