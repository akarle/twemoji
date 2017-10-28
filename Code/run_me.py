# Imports
from load_data import load_data
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import numpy as np

# Load Data
data_path = os.path.join('..','Data','trial')
label_path = os.path.join(data_path, 'us_trial.labels')
text_path = os.path.join(data_path, 'us_trial.text')

data, labels = load_data(text_path, label_path)


# Extract Features
#TODO: use feature extractor class... just throwing this in for an ex
count_vect = CountVectorizer()
x_counts = count_vect.fit_transform(data)
print x_counts.shape
# print count_vect.vocabulary_


# Instantiate Classifiers
clf = MultinomialNB()


# Train Classifiers on Extracted Features
clf.fit(x_counts, labels)

# Evaluate Classifiers
scores = cross_val_score(clf, x_counts, labels, cv=5)
print 'average accuracy score: ', np.mean(scores)

# Graphing