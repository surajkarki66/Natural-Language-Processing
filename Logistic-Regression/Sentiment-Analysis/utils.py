import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickets like $GE
    tweet = re.sub(r'\$\w*', ' ', tweet)
    # remove old style retweet text RT
    tweet = re.sub(r'^RT[\s]+', ' ', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', ' ', tweet)
    # remove hash tags
    tweet = re.sub(r'#', '', tweet)
    # instantiate tokenizer class
    tokenizer = TweetTokenizer(
        preserve_case=False, strip_handles=True, reduce_len=True)
    # tokenize tweets
    tweet_tokens = tokenizer.tokenize(tweet)

    clean_tweets = []
    for word in tweet_tokens:
        if (word not in stopwords_english) and (word not in string.punctuation):
            # clean_tweets.append(word)
            stem_word = stemmer.stem(word)
            clean_tweets.append(stem_word)
    return clean_tweets
