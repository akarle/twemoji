""" File to load in the data

    Loads the tweets into a list of strings `data`
    Loads the labels into a seperate list, `labels`
"""

import numpy as np
import os

data_path = os.path.join('..','Data','trial')

label_file = os.path.join(data_path, 'us_trial.labels')
data_file = os.path.join(data_path, 'us_trial.text')

# load in data
lf = open(label_file, 'r')
tf = open(data_file, 'r')

data = []
labels = []


for tweet in tf:
    label = lf.readline().rstrip() #rstrip to remove trailing \n
    data.append(tweet.rstrip())
    labels.append(label)

tf.close()
lf.close()

print 'First 10 tweets: ', data[:10]
print 'First 10 labels: ', labels[:10]

# Brief count vector example:
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
x_counts = count_vect.fit_transform(data)
print x_counts.shape