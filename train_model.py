#Tomes, Christopher
#Cal Poly Pomona CS4650
#train_model.py
#This program will make a prediction on which stocks to buy and sell.
#
import json
import pandas as pd
from sklearn.linear_model import LinearRegression
import datetime
import os

def datetime_to_seconds(df, col_name):
    """
    Converts a datetime column in a Pandas DataFrame to the number of seconds since midnight.
    :param df: Pandas DataFrame containing datetime column
    :param col_name: Name of the datetime column in the DataFrame
    :return: Pandas Series containing the number of seconds since midnight for each datetime value in the DataFrame
    """
    # Extract the datetime column and convert it to datetime type if it's not already
    datetime_col = pd.to_datetime(df[col_name])

    # Convert datetime values to the number of seconds since midnight
    seconds_since_midnight = (datetime_col.dt.hour * 3600) + (datetime_col.dt.minute * 60) + datetime_col.dt.second

    return seconds_since_midnight
# Get the path to the data directory
data_dir = './data/'

# Get a list of all JSON files in the data directory
json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

conversion = []
best_meta = 0
total_conversion = 0.0
# Loop through the JSON files and load the data from each file
for json_file in json_files:
    with open(data_dir + json_file, 'r') as f:
        data = json.load(f)
        # Do something with the data
# Load the JSON data from a file
#with open('./data/AMZN.json', 'r') as f:
#    data = json.load(f)
#print(data)
# Access the 'meta' and 'values' keys in the loaded JSON data
    meta_data = data['meta']
    values_data = data['values']
# Convert the data to a Pandas DataFrame
    print(meta_data)
    df = pd.DataFrame(values_data)
    df['next_close'] = df['close'].shift(+1)

# Convert datetime column to datetime type
    df['datetime'] = pd.to_datetime(df['datetime'])

# Convert all other columns to numeric type
    df[['open', 'high', 'low', 'close', 'volume', 'next_close']] = df[['open', 'high', 'low', 'close', 'volume', 'next_close']].apply(pd.to_numeric)
    df['time'] = df['datetime'].dt.time

    df['time'] = datetime_to_seconds(df, 'datetime')
    predicted_class= df.iloc[0]
    df.drop(df.head(1).index, inplace=True)

    print(predicted_class)
# Create a new dataframe with the closing prices and the next closing price
# extract the time component
    #print(df)
    X = df[['time','open','close','volume']]
    y = df['next_close']
    print('beginning LR')
    model = LinearRegression()
    model.fit(X, y)
# Predict the next closing price based on the most recent closing price
    last_close = last_close = [predicted_class['time'],predicted_class['open'], predicted_class['close'], predicted_class['volume']]



    next_close = model.predict([last_close])
    print('Predicted next closing price:',next_close[0] )
    print('Current price:',last_close[2] )
    print((next_close[0]/last_close[2]), ' of my money is expected to exist after trade')
    total_conversion= total_conversion + (next_close[0]/last_close[2])
    conversion.append([meta_data['symbol'], (next_close[0]/last_close[2]), last_close[2]])
    print('Printtest',last_close)
total_conversion=0
print(total_conversion)
for conv in conversion:
    print(conv)
    if conv[1] > 1.0:
        total_conversion+= conv[1]
print(total_conversion)
print('My pie chart of assets should be:')
for conv in conversion:
    if conv[1] < 1:
        continue
    conv[1]/=total_conversion
    print('BUY: ',conv[0:2], 'Buy', (int)(conv[1]*1000000/conv[2]), 'shares.')

