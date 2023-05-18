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
