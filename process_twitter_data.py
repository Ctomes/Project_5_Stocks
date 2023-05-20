import pandas as pd
import math
import numpy as np
import csv
import os
def engagement_metric(row):
    val = row['Like Count'] 
    newval = math.log2(val)
    return newval
# Assuming you have a DataFrame called 'df' with columns 'column1', 'column2', etc.
# Load the CSV file into a DataFrame
df = pd.read_csv('tweets.csv')
columns_to_remove = ['Tweet Id', 'Text', 'Quote Count','Vibe', 'Retweet Count', 'Conversation Id', 'Reply Count']
# Drop the unnamed first column
df = df.iloc[:, 1:]
df = df.drop(columns_to_remove, axis=1)
# Get the unique values from the first column
unique_values = df['STOCK'].unique()

# Create a dictionary to store the split DataFrames
dfs = {}

# Split the DataFrame based on unique values in the first column
for value in unique_values:
    dfs[value] = df[df['STOCK'] == value]
    # Calculate the running total using cumulative sum
# Access the split DataFrames using the unique values
for value, split_df in dfs.items():
    split_df['date'] = pd.to_datetime(split_df['Datetime']).dt.date
    split_df = split_df.dropna()
    #split_df = split_df.groupby('date').sum('Like Count')
    #print(grouped_df)
    split_df['Engagement Score'] = split_df['Like Count']/split_df['View Count']
    split_df = split_df.groupby('date').sum('Like Count')
    split_df['Engagement Score'] = split_df['Engagement Score'] - split_df['Engagement Score'].median()
    print(f"DataFrame for {value}:")
    print(split_df)
    filename = (f"{value}.csv")
    filepath = os.path.join("data_twitter2", filename)
    split_df.to_csv(filepath)
    print()

