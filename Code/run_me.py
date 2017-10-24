# Imports
from load_data import load_data
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
import argparse

# Parse arguments
parser = argparse.ArgumentParser(
    description="Run emoji prediction classifiers and output accuracy results.")
parser.add_argument('-c', '--classifier', nargs='+', default='all', choices=['nb', 'lr'],
    help='specifies which classifier(s) to use (default: %(default)s) possible classifiers: %(choices)s',
    metavar='C', dest='classifier_type')
args = parser.parse_args()
print(args)

# Helper functions
def baseline_predict(labels):
    """ A function that predicts the most common label """
    #get the most common label:
    labels_array = np.array(labels)
    mc_label = np.argmax(np.bincount(labels_array))
    print "Most common label is ", mc_label

    #return an array of the most common label (one per data case)
    preds = np.multiply(mc_label, np.ones(len(labels)))
    print preds
    return preds


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
print 'Average accuracy score for NB with unigrams: ', np.mean(scores)

baseline_score = accuracy_score(labels, baseline_predict(labels))
print "Baseline accuracy score: ", baseline_score

# Graphing
