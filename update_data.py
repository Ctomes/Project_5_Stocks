#Tomes, Christopher
#Cal Poly Pomona CS4650
#update_data.py
#This program will make a prediction on which stocks to buy and sell.
#
import requests
import json
import os
#Grab api key from folder
def read_api_key_file(filename):
    with open(filename, 'r') as file:
        api_key = file.readline().strip()
    return api_key
#Grab Data from api.
def get_time_series(ticker_symbol, api, start_date, interval):
    url = f"https://api.twelvedata.com/time_series?symbol={ticker_symbol}&interval={interval}&format=JSON&start_date={start_date}&apikey={api}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        filename = f"{ticker_symbol}.json"
        filepath = os.path.join("data3", filename)
        with open(filepath, "w") as f:
            json.dump(data, f)
        print(f"JSON data saved to file {filepath}!")
    else:
        print("Request failed with status code:", response.status_code)
    return response


api_key = read_api_key_file("api_key.txt")
start_date = "04/01/2023 8:00 PM"
interval = "1day"
tickers = {"AAPL","AMZN","GOOGL","MSFT","NFLX","TSLA","NVDA","INTC"}


print('Downloading Dat...')
for ticker in tickers:
    get_time_series(ticker, api_key, start_date, interval)

print('Done!')

