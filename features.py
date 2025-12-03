from preprocess import process_tweet
import numpy as np

def extract_features(tweet, freqs, process_tweet = process_tweet):
    word_l = process_tweet(tweet)

    x = np.zeros(3)
    x[0] = 1
    for word in word_l:
        x[1] += freqs.get((word, 1.0), 0)
        x[2] += freqs.get((word, 0.0), 0)
    x = x[None, :]
    assert (x.shape == (1, 3))
    return x