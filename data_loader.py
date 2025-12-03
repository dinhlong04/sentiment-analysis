# data_loader.py
import numpy as np
import nltk
from nltk.corpus import twitter_samples
from config import Config

def ensure_nltk_resources():
    for resource in Config.NLTK_DATA:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=True)

def load_data():
    ensure_nltk_resources()
    all_positive = twitter_samples.strings('positive_tweets.json')
    all_negative = twitter_samples.strings('negative_tweets.json')
    return all_positive, all_negative

def prepare_data(pos_tweets, neg_tweets):
    # Chia train/test theo Config
    split_idx = Config.TEST_SPLIT_INDEX
    
    train_pos = pos_tweets[:split_idx]
    test_pos = pos_tweets[split_idx:]
    train_neg = neg_tweets[:split_idx]
    test_neg = neg_tweets[split_idx:]

    train_x = train_pos + train_neg
    test_x = test_pos + test_neg

    # Tạo labels
    train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
    test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

    return train_x, test_x, train_y, test_y