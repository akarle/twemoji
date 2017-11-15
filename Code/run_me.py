# Imports
from plotting import acc_bar_chart, plot_confusion_matrix
from load_data import load_data
from text_feat_extractor import TextFeatureExtractor
from feature_combinator import FeatureCombinator
import os
import fnmatch
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from baseline import baseline_predict
import argparse
import random
from parse_loadout import parse_loadout as pl
import time
import re
from pos_tagger import pos_tag
import unicodedata
if os.name != 'nt':
    from spell_checker import correct_spelling

# ##############################################
#               PARSE ARGUMENTS
# ##############################################

classops = ['nb', 'lr', 'svm']
parser = argparse.ArgumentParser(
    description="Run emoji prediction classifiers and output results.")

parser.add_argument('-c', '--classifier',
                    nargs='+',
                    default=classops,
                    choices=classops,
                    help='specifies which classifier(s) to use (default: %(default)s)\
                          possible classifiers: %(choices)s',
                    metavar='C',
                    dest='classifier_type')

parser.add_argument('-d', '--data',
                    nargs=1,
                    required=True,
                    help='name of dataset subdirectory to use in Data directory \
                          (must be in Data directory)',
                    metavar='dataset',
                    dest='data')

parser.add_argument('-n',
                    nargs=1,
                    type=int,
                    help='number of data entries to train/evaluate on',
                    metavar='N',
                    dest='num_instances')

parser.add_argument('-v', '--verbose',
                    action='count',
                    default=1,
                    help='set verbosity flag',
                    dest='verbose')

parser.add_argument('-s', '--silent',
                    action='store_const',
                    const=0,
                    help='set verbosity flag to silent',
                    dest='verbose')

parser.add_argument('-l', '--loadout',
                    nargs=1,
                    default=None,
                    help='name of loadout JSON file in Loadouts directiory',
                    metavar='loadout',
                    dest='loadout')

args = parser.parse_args()

if args.verbose >= 1:
    def verboseprint(*args):
        for arg in args:
            print arg,
        print
else:
    verboseprint = lambda *a: None

cverbosity = args.verbose >= 2

if args.loadout is not None:
    cl, pre, cvargs, ftyps, desc = pl(args.loadout[0])
else:
    cl = args.classifier_type
    pre = None
    cvargs = None
    ftyps = ['unigram']
    desc = ""

verboseprint("*******")
runstr = "Running " + str(cl) + \
    " classifier(s) on dataset " + args.data[0]
if args.num_instances:
    runstr += " for " + str(args.num_instances[0]) + " tweets"
verboseprint(runstr)
verboseprint("*******")


# ##############################################
#                   LOAD DATA
# ##############################################

data_path = os.path.join('..', 'Data', args.data[0])
if not os.path.isdir(data_path):
    raise Exception('Your specified data directory ' +
                    data_path + ' does not exist.')

label_path = None
text_path = None
for f in os.listdir(data_path):
    if fnmatch.fnmatch(f, '*.labels') and label_path is None:
        label_path = os.path.join(data_path, f)
    elif fnmatch.fnmatch(f, '*.text') and text_path is None:
        text_path = os.path.join(data_path, f)
if label_path is None:
    raise Exception('Could not find a labels file.')
if text_path is None:
    raise Exception('Could not find a text file.')

if args.num_instances:
    data, labels, dcount = load_data(text_path,
                                     label_path, args.num_instances[0])
else:
    data, labels, dcount = load_data(text_path, label_path)

# Randomize data order to prevent overfitting to subset of
# data when running on fewer instances
combined = list(zip(data, labels))
random.shuffle(combined)
data[:], labels[:] = zip(*combined)

verboseprint("Loaded ", dcount, " tweets...")
verboseprint('First 10 tweets and labels: ')
verboseprint("|   Label ::: Tweet")
verboseprint("|   ---------------")

for i in range(10):
    verboseprint('|%6s' % labels[i], " ::: ", data[i])
verboseprint("*******")


# ##############################################
#             MANUAL PREPROCESSING
# ##############################################

# SPELL CORRECTION
if os.name != 'nt':
    if 'spell-correction' in pre:
        verboseprint("Spell correcting tweets....")
        data = [correct_spelling(tweet) for tweet in data]
        verboseprint("Done spell correction")

# WORD REPLACEMENT USING CLUSTERS
if 'word-clustering' in pre:
    verboseprint("Word replacement using clusters....")
    clusters_path = os.path.join('..', 'Data', '50mpaths2.txt')
    clusters = open(clusters_path, "r")
    dict = {}

    for line in clusters:
        temp = re.split(r'\t+', line)
        dict[temp[1]] = temp[0]

    data_temp = []
    for line in data:
        temp = line.split(' ')
        for x in temp:
            if x in dict:
                y = list(x)
                escaped = ""
                for c in y:
                    if c in ["-", "[", "]", "\\", "^", "$", "*", ".", "+",
                             ")", "(", "?", "|", "{", "}"]:
                        escaped += ("\\"+c)
                    else:
                        escaped += c
                line = re.sub(r"\b%s\b" % escaped, dict[x], line)
        data_temp.append(line)
    data = data_temp
    dict.clear()
    clusters.close()

# POS TAGGING
if 'pos-tags' in pre:
    verboseprint("Adding POS tags to tweets...")
    data = pos_tag([u"".join([c for c in unicodedata.normalize(
        'NFKD', unicode(d, 'utf8'))
        if not unicodedata.combining(c)]) for d in data])
    verboseprint("Finished adding POS tags")
    verboseprint("*******")

# ##############################################
#               EXTRACT FEATURES
# ##############################################

verboseprint("Extracting features...")
extractor = TextFeatureExtractor()
feats = extractor.extract_features(data, ftyps, cvargs)

# TODO : reconnect the sentiment classifiers! Load in from pickle!

# Use Combinator to Combine Features
combinator = FeatureCombinator(feats)


# ##############################################
#           INSTANTIATE CLASSIFIERS
# ##############################################

clfs = {}
tick_names = []  # seperate for graph tick labels
if 'nb' in cl:
    tick_names.append('Multi. NB')
    clfs['Multinomial Naive Bayes'] = MultinomialNB()
if 'lr' in cl:
    tick_names.append('LogReg')
    clfs['Logistic Regression'] = LogisticRegression(verbose=cverbosity)
if 'svm' in cl:
    tick_names.append('Lin. SVM')
    clfs['Linear SVM'] = LinearSVC(verbose=cverbosity)

# ##############################################
#           TRAIN AND EVALUATE CLFS
# ##############################################

# Dict of {Classifer : [(feat_combo, score)]}
# That is, classifer mapped to a list of tuples of (feat_combo, score)
scores = {}

# Train Classifiers on Extracted Features
# Use FeatureCombinator to loop through all combos
feat_perm = combinator.next_perm()

while feat_perm is not None:
    verboseprint("Current feat_perm: ", feat_perm[0])
    verboseprint("Features Shape: ", feat_perm[1].shape)
    verboseprint("******")

    # Split data into train and test:
    X_train, X_test, y_train, y_test = train_test_split(
        feat_perm[1], labels, test_size=0.30, random_state=0)

    for c in clfs:
        # Train (and Tune Hyperparams)
        clfs[c].fit(X_train, y_train)

        # Score, save score
        preds = clfs[c].predict(X_test)
        score = accuracy_score(y_test, preds)

        verboseprint("Classifer: ", c, " with features: ", feat_perm[0])
        unique, counts = np.unique(preds, return_counts=True)
        verboseprint('Predicted label counts: \n', dict(zip(unique, counts)))
        unique, counts = np.unique(y_test, return_counts=True)
        verboseprint('Gold label counts: \n', dict(zip(unique, counts)))
        prf1 = precision_recall_fscore_support(y_test,
                                               preds, average='weighted')
        verboseprint('Weighted precision:', prf1[0])
        verboseprint('Weighted recall:', prf1[1])
        verboseprint('Weighted f1_score:', prf1[2])
        if c not in scores:
            scores[c] = []

        cm = confusion_matrix(y_test, preds, labels=np.arange(10))
        scores[c].append((str(feat_perm[0]), score, cm))

        verboseprint("Average accuracy score: %f"
                     % (score,))
        verboseprint("*******")

    # Go to next feature permutation
    feat_perm = combinator.next_perm()


# Baseline score
verboseprint("*******")
verboseprint("Calculating baseline...")
mc_label, baseline_preds = baseline_predict(y_test)
baseline_score = accuracy_score(y_test, baseline_preds)
verboseprint("Most common label is: ", mc_label)
verboseprint("Baseline accuracy score: ", baseline_score)
verboseprint("*******")

# Print comparison table
print 'Summary:'
print "*" * 92
print '*', '%-70s' % ('Classifier',), '|', '%-15s' % ('Score',), '*'
print '*', '-' * 88, '*'
for c in scores:
    print '*', '%-70s' % (c,), '|', '%-15s' % ("",), '*'
    for fcombo, score, _ in scores[c]:
        print '*', '     %-65s' % (fcombo,), '|', \
              '%-15s' % (str(score),), '*'
    print "*" * 92

# ##############################################
#               GRAPH EVALUATIONS
# ##############################################

# TODO: construct the output file based on the parameters!
for c in scores:
    output_file = '../Figures/' + c + \
                  '_out_' + time.strftime("%Y%m%d-%H%M%S") + '.png'
    labels = []
    values = []
    cms = []
    for label, value, cm in scores[c]:
        labels.append(label)
        values.append(value)
        cms.append(cm)

    acc_bar_chart(
        c + " (n=" + str(dcount) + ")",
        desc,
        baseline_score,
        values,
        labels,
        output_file
    )

    # Find best confusion matrix
    maxind = np.argmax(values)
    max_cm = cms[maxind]
    max_label = labels[maxind]

    conf_file = '../Figures/' + c + 'CONF_MTX_' +\
                time.strftime("%Y%m%d-%H%M%S") + '.png'

    plot_confusion_matrix(max_cm, c, max_label, conf_file)
