#Tomes, Christopher
#Cal Poly Pomona CS4650
#stocks_proj.py
#This program will make a prediction on which stocks to buy and sell.
#
import requests
import json
import time
import subprocess



#Data mine some values.
#Avoid large block ownership.
#Tesla is an example of the oppisite.
#There is a BID-ASK-SPREAD

#Im thinking tech sector. Gigabyte, Intel, AMD, NVidia, EVGA, ASUS, SAMSUNG, LG etc

#moved to update_model.py
print('Downloading Dat...')
subprocess.call(['python', 'train_model.py'])

print('Training Data..')

subprocess.call(['python', 'train_model.py'])

print('Make Prediction:')
