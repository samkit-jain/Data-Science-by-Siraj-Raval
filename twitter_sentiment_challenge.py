import os
import csv
import tweepy
from textblob import TextBlob

consumer_key = os.environ.get('CK', None)
consumer_secret = os.environ.get('CS', None)

access_token = os.environ.get('AT', None)
access_token_secret = os.environ.get('ATS', None)

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

fieldnames = ['Tweet', 'Polarity']
polarities = ['Positive', 'Negative']

writer = csv.DictWriter(open("output.csv", "w"), fieldnames=fieldnames)
writer.writeheader()

public_tweets = api.search('Trump')

for tweet in public_tweets:
	analysis = TextBlob(tweet.text)

	writer.writerow({
		"Tweet": str(tweet.text),
		"Polarity": polarities[int(analysis.sentiment.polarity < 0)]
	})
