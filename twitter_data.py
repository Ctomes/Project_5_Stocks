import tweepy
import pandas as pd

auth = tweepy.OAuth2BearerHandler("Bearer Token here")
api = tweepy.API(auth)
# Get the User object that represents the user, @Twitter
user = api.get_user(screen_name="Twitter")

print(user.screen_name)
print(user.followers_count)
for friend in user.friends():
   print(friend.screen_name)      