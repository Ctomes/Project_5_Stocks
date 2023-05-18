# importing libraries and packages
import snscrape.modules.twitter as sntwitter
import pandas as pd

stocks = {"AAPL": "Apple","AMZN": "amazon","GOOGL": "Google","MSFT": "Microsoft","NFLX": "netflix","TSLA": "Tesla","NVDA": "nvidia","INTC" : "intel"}
# Creating list to append tweet data 
tweets_list1 = []
for stock in stocks:
   print(stocks[stock])
   if stocks[stock] == 'Apple':
      #Apple doesn't tweet, they will recieve a default value for now.
      #Possibly enumerate through their satalite accounts like Apple Music, Apple Support and attribute to Apple in Future
      continue

# Using TwitterSearchScraper to scrape data and append tweets to list
   for i,tweet in enumerate(sntwitter.TwitterSearchScraper('from:'+stocks[stock]).get_items()): #declare a username 
      if i>100: #number of tweets you want to scrape
        break
      tweets_list1.append([stock,tweet.date, tweet.id, tweet.rawContent, tweet.replyCount, tweet.likeCount, tweet.quoteCount, tweet.viewCount, tweet.vibe, tweet.retweetCount, tweet.conversationId]) 
    #declare the attributes to be returned
    
# Creating a dataframe from the tweets list above 
tweets_df1 = pd.DataFrame(tweets_list1, columns=['STOCK', 'Datetime', 'Tweet Id', 'Text', 'Reply Count', 'Like Count', 'Quote Count', 'View Count', 'Vibe', 'Retweet Count', "Conversation Id"])

filename = "tweets.csv"
tweets_df1.to_csv(filename)
