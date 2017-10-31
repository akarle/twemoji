# Imports
from load_data import load_data
import feature_extractor
import os
import fnmatch
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
import argparse

# Parse arguments
classops = ['nb', 'lr', 'svm']
parser = argparse.ArgumentParser(
    description="Run emoji prediction classifiers and output accuracy results.")
parser.add_argument('-c', '--classifier', nargs='+', default=classops, choices=classops,
    help='specifies which classifier(s) to use (default: %(default)s) possible classifiers: %(choices)s',
    metavar='C', dest='classifier_type')
parser.add_argument('-d', '--data', nargs=1, required=True,
    help='name of dataset subdirectory to use in Data directory (must be in Data directory)',
    metavar='dataset', dest='data')
parser.add_argument('-n', nargs=1, type=int,
    help='number of data entries to train/evaluate on',
    metavar='N', dest='num_instances')
parser.add_argument('-v', '--verbose', action='count', default=1,
    help='set verbosity flag',
    dest='verbose')
parser.add_argument('-s', '--silent', action='store_const', const=0,
    help='set verbosity flag to silent',
    dest='verbose')
args = parser.parse_args()

if args.verbose >= 1:
    def verboseprint(*args):
        for arg in args:
           print arg,
        print
else:
    verboseprint = lambda *a: None

cverbosity = args.verbose >= 2

verboseprint("*******")
runstr = "Running " + str(args.classifier_type) + " classifier(s) on dataset " + args.data[0]
if args.num_instances:
    runstr += " for " + str(args.num_instances[0]) + " tweets"
verboseprint(runstr)
verboseprint("*******")

# Helper functions
def baseline_predict(labels):
    """ A function that predicts the most common label """
    #get the most common label:
    labels_array = np.array(labels)
    mc_label = np.argmax(np.bincount(labels_array))
    verboseprint("Most common label is ", mc_label)

    #return an array of the most common label (one per data case)
    preds = np.multiply(mc_label, np.ones(len(labels)))
    #print preds
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

if args.num_instances:
    data, labels = load_data(text_path, label_path, args.num_instances[0], verbose=args.verbose)
else:
    data, labels = load_data(text_path, label_path, verbose=args.verbose)


# Extract Features
verboseprint("Extracting features...")
extractor = feature_extractor.FeatureExtractor()
x_counts = extractor.extract_features(data, ['unary', ('sent_analysis', 'baseline')])
verboseprint("Features shape: ", x_counts.shape)
# print count_vect.vocabulary_
verboseprint("*******")

# Baseline score
verboseprint("Calculating baseline...")
baseline_score = accuracy_score(labels, baseline_predict(labels))
verboseprint("Baseline accuracy score: ", baseline_score)
verboseprint("*******")

# Instantiate Classifiers
clfs = {}
if 'nb' in args.classifier_type:
    clfs['<Multinomial Naive Bayes>'] = MultinomialNB()
if 'lr' in args.classifier_type:
    clfs['<Logistic Regression>'] = LogisticRegression(verbose=cverbosity)
if 'svm' in args.classifier_type:
    clfs['<Linear SVM>'] = LinearSVC(verbose=cverbosity)

# Train Classifiers on Extracted Features
averages = {}
for c in clfs:
    clfs[c].fit(x_counts, labels)
    # Evaluate Classifier
    scores = cross_val_score(clfs[c], x_counts, labels, cv=5)
    averages[c] = np.mean(scores)
    verboseprint('Average accuracy score for', c, 'with unigrams: ', np.mean(scores))
    verboseprint("*******")

# Print comparison table
if len(averages) > 1 or args.verbose == 0:
    print 'Summary:'
    print "**************************************************************"
    print '*', '%-40s' % ('Classifier',), '|', '%-15s' % ('Score',), '*'
    print '*', '---------------------------------------------------------- *'
    for c in averages:
        print '*', '%-40s' % (c,), '|', '%-15s' % (str(averages[c]),), '*'
    print "**************************************************************"

# Graphing
