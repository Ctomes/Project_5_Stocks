import pandas as pd
from pytrends.request import TrendReq
from sklearn.linear_model import LinearRegression
from datetime import datetime
import time

def predict_interest(keyword, starttime):
    time.sleep(1)
    prediction ={
                 "current_trend": 1.0,
                 "predicted_trend": 1.0,
                 "delta_trend": 0}
    
    if keyword == 'NA':
        return prediction
    
    # Set up Google Trends API
    pytrends = TrendReq(hl='en-US', tz=360)
    
    timeframe = starttime
    # Query Google Trends for interest over time
    pytrends.build_payload(kw_list=[keyword], timeframe=timeframe)
    interest_over_time = pytrends.interest_over_time()
    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(interest_over_time)
    df = df.drop(df.index[-1])

    return df[keyword]
    # Extract relevant features for linear regression
    X = df.index.astype('int64').values.reshape(-1, 1)
    X = X // 10**9
    y = df[keyword].values
    prediction['current_trend'] = y[-1]

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict interest for the next hour
    next_hour = int(df.index[-1].timestamp() + 3600)  # Add 3600 seconds for next hour
    predicted_interest = model.predict([[next_hour]])

    prediction['predicted_trend'] = predicted_interest[0]
    prediction['delta_trend'] = prediction['predicted_trend']-prediction['current_trend']

    return prediction



#stocks = ['GOOGL', 'AAPL', 'AMZN', 'INTC', 
#         'MSFT', 'NTFLX', 'NVDA', 'TSLA']
#stocks = ['GOOGL']
#predictions = {}

#for stock in stocks:
#    print("Predicting Interest for:", stock)
#    predictions[stock] = predict_interest(stock,['today 1-m'])
#    time.sleep(4)

#print(predictions)

