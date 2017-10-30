# Imports
from load_data import load_data
import os
import fnmatch
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
parser.add_argument('-d', '--data', nargs=1, required=True,
    help='name of dataset subdirectory to use in Data directory (must be in Data directory)',
    metavar='dataset', dest='data')
parser.add_argument('-n', nargs=1,
    help='number of data entries to train/evaluate on',
    metavar='N', dest='num_instances')
args = parser.parse_args()

print "*******"
print "Running " + str(args.classifier_type) + " classifier(s) on dataset " + args.data[0]
print "*******"

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
data_path = os.path.join('..','Data', args.data[0])
if not os.path.isdir(data_path):
    raise Exception('Your specified data directory ' + data_path + ' does not exist.')
label_path = None
text_path = None
for f in os.listdir(data_path):
    if fnmatch.fnmatch(f, '*.labels') and label_path == None:
        label_path = os.path.join(data_path, f)
    elif fnmatch.fnmatch(f, '*.text') and text_path == None:
        text_path = os.path.join(data_path, 'us_trial.text')
if label_path == None:
    raise Exception('Could not find a labels file.')
if text_path == None:
    raise Exception('Could not find a text file.')

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
