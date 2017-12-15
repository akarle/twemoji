# Imports
from plotting import acc_bar_chart, plot_confusion_matrix
from load_data import load_emoji, load_sentiment
from text_feat_extractor import TextFeatureExtractor
from feature_combinator import FeatureCombinator
import os
import fnmatch
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
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
import pickle
import unicodedata
from sent_predictor import predict_sent
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

parser.add_argument('-p', '--pipeline',
                    nargs=1,
                    required=True,
                    help='state whether you are training sentiment classifiers \
                          or training emoji classifiers',
                    metavar='pipeline',
                    dest='pipeline')

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

parser.add_argument('-nf', '--nofigures',
                    action='store_true',
                    help='a flag that when added prevents saving of figures',
                    dest='nofigs')

args = parser.parse_args()

if args.verbose >= 1:
    def verboseprint(*args):
        for arg in args:
            print arg,
        print
else:
    verboseprint = lambda *a: None  # noqa

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

if args.pipeline[0] == 'emoji':
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
        data, labels, dcount = load_emoji(text_path,
                                          label_path, args.num_instances[0])
    else:
        data, labels, dcount = load_emoji(text_path, label_path)

elif args.pipeline[0] == 'sent':
    if args.num_instances:
        data, labels, dcount = \
            load_sentiment(data_path, num_instances=args.num_instances[0])
    else:
        data, labels, dcount = load_sentiment(data_path)
else:
    raise Exception('Invalid pipeline choice: choose either `emoji` or `sent`')

# Randomize data order to prevent overfitting to subset of
# data when running on fewer instances
combined = list(zip(data, labels))
random.shuffle(combined)
data[:], labels[:] = zip(*combined)

# Label remap
if 'remap' in pre and args.pipeline[0] == 'emoji':
    verboseprint("Remapping labels....")
    templabel = []
    for label in labels:
        if label in [0,3,8,13]:
            templabel.append(0)
        if label in [1,9]:
            templabel.append(11)
        elif label in [2]:
            templabel.append(1)
        elif label in [4]:
            templabel.append(2)
        elif label in [5,6,16]:
            templabel.append(3)
        elif label in [7]:
            templabel.append(4)
        elif label in [10,18]:
            templabel.append(5)
        elif label in [11]:
            templabel.append(6)
        elif label in [12]:
            templabel.append(7)
        elif label in [14,19]:
            templabel.append(8)
        elif label in [15]:
            templabel.append(9)
        elif label in [17]:
            templabel.append(10)  
    labels = templabel

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
        data = correct_spelling(data)
        verboseprint("Done spell correction")

data_for_sent = data

# POS TAGGING
if 'pos-tags' in pre:
    verboseprint("Adding POS tags to tweets...")
    data = pos_tag([u"".join([c for c in unicodedata.normalize(
        'NFKD', unicode(d, 'utf8'))
        if not unicodedata.combining(c)]) for d in data])
    verboseprint("Finished adding POS tags")
    verboseprint("*******")

# WORD REPLACEMENT USING CLUSTERS
if 'word-clustering' or 'word-clustering-append' in pre:
    clusters_path = os.path.join('..', 'Data', '50mpaths2.txt')
    clusters = open(clusters_path, "r")
    clusterdict = {}

    for line in clusters:
        temp = re.split(r'\t+', line)
        clusterdict[temp[1]] = temp[0]

    if 'word-clustering' in pre:
        verboseprint("Word replacement using clusters....")
        data_temp = []
        for line in data:
            temp = line.split(' ')
            for x in temp:
                if 'pos-tags' in pre:
                    x = x.split('_')[2]
                if x in clusterdict:
                    y = list(x)
                    escaped = ""
                    for c in y:
                        if c in ["-", "[", "]", "\\", "^", "$", "*", ".", "+",
                                 ")", "(", "?", "|", "{", "}"]:
                            escaped += ("\\" + c)
                        else:
                            escaped += c
                    if 'pos-tags' in pre:
                        line = re.sub(r"(?<=POS_._)(\W\B|[\W]*[\w]+\b)"
                                      % escaped.lower(), clusterdict[x], line)
                    else:
                        line = re.sub(r"\b%s\b" % escaped.lower(),
                                      clusterdict[x], line)
            data_temp.append(line)
        data = data_temp

    if 'word-clustering-append' in pre:
        verboseprint("Word appending using clusters....")
        data_temp = []
        for line in data:
            temp = line.split(' ')
            for x in temp:
                if 'pos-tags' in pre:
                    x = x.split('_')[2]
                if x in clusterdict:
                    line = line+" "+clusterdict[x]
            data_temp.append(line)
        data = data_temp

    clusterdict.clear()
    clusters.close()

# ##############################################
#               EXTRACT FEATURES
# ##############################################

clf_feats = {}

if 'sent-class' in ftyps:
    ftyps.remove('sent-class')
    verboseprint("*******")
    verboseprint("Loading pickled sentiment classifiers...")
    with open('sent.pkl', 'r') as f:
        clf, clf_name, pickleables, perm = pickle.load(f)

    verboseprint("The best sent_clf stored is: %s (params: %s) with feats %s"
                 % (clf_name, clf.get_params(), perm))

    clf_feats = predict_sent(data_for_sent, clf, clf_name, pickleables, perm)
    preds = clf_feats[clf_name]
    print "num non neutral preds: ", (len(preds[preds != 1]), len(preds))
    verboseprint("Sentiment prediction features complete")
    verboseprint("*******")

verboseprint("Extracting text features...")
extractor = TextFeatureExtractor()
feats = extractor.extract_features(data, ftyps, cvargs)


# Use Combinator to Combine Features
combinator = FeatureCombinator(feats, clf_feats)


# ##############################################
#           INSTANTIATE CLASSIFIERS
# ##############################################

clfs = {}
tick_names = []  # seperate for graph tick labels
if 'nb' in cl:
    tick_names.append('Multi. NB')
    clfs['Multinomial Naive Bayes'] = [MultinomialNB(), {}]
if 'lr' in cl:
    tick_names.append('LogReg')
    clfs['Logistic Regression'] = [LogisticRegression(verbose=cverbosity), {}]
if 'svm' in cl:
    tick_names.append('Lin. SVM')
    clfs['Linear SVM'] = [LinearSVC(verbose=cverbosity), {}]
if 'rf' in cl:
    tick_names.append('RandForest')
    hyp = {'n_estimators': [10, 30, 50],
           'max_depth': [None, 1, 3, 10]}
    clfs['RF'] = [RandomForestClassifier(n_jobs=-1, verbose=cverbosity), hyp]

# ##############################################
#           TRAIN AND EVALUATE CLFS
# ##############################################

# Dict of {Classifer : [(feat_combo, score)]}
# That is, classifer mapped to a list of tuples of (feat_combo, score)
scores = {}

# The best so far (to be stored if training sent clf)
best = {'clf': None, 'perm': None, 'score': 0}

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
        hypdict = clfs[c][1]
        if not hypdict:  # empty dict
            clfs[c][0].fit(X_train, y_train)
            bestclf = clfs[c][0]
            bestparams = {}

        else:
            # Cross val over hyperparams!
            gridcv = GridSearchCV(clfs[c][0], hypdict, verbose=cverbosity)
            gridcv.fit(X_train, y_train)
            bestclf = gridcv
            bestparams = gridcv.best_params_

        # Score, save score
        preds = bestclf.predict(X_test)
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

        cm = confusion_matrix(y_test, preds,
                              labels=np.arange(np.max(labels) + 1))
        scores[c].append((str(feat_perm[0]), score, cm))

        verboseprint("Average accuracy score: %f"
                     % (score,))
        verboseprint("*******")

        # Add best clf to list of clfs for pickle
        if score > best['score']:
            print "Found new best!"
            best['score'] = score
            best['clf'] = c
            best['perm'] = feat_perm
            best['params'] = bestparams

        verboseprint("%s: (params: %s)" % (c, bestparams))
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
    graph_labels = []
    values = []
    cms = []
    for label, value, cm in scores[c]:
        graph_labels.append(label)
        values.append(value)
        cms.append(cm)

    acc_bar_chart(
        c + " (n=" + str(dcount) + ")",
        desc,
        baseline_score,
        values,
        graph_labels,
        output_file,
        args.nofigs
    )

    # Find best confusion matrix
    maxind = np.argmax(values)
    max_cm = cms[maxind]
    max_label = graph_labels[maxind]

    conf_file = '../Figures/' + c + 'CONF_MTX_' +\
                time.strftime("%Y%m%d-%H%M%S") + '.png'

    plot_confusion_matrix(max_cm, c, max_label, conf_file,
                          args.pipeline[0], args.nofigs)

# ##############################################
#            SAVE TOP CLFS (SENT ONLY)
# ##############################################

if args.pipeline[0] == 'sent':
    verboseprint("Saving the top clf to pickle dump")
    # Pull best from evaluation from curr_best
    verboseprint("Best overall: %s (params: %s) with %s giving score %f" %
                 (best['clf'], best['params'], best['perm'][0], best['score']))

    # Retrain the model with the data
    if best['params']:
        print best['params']
        clfs[best['clf']][0].set_params(**best['params'])

    clfs[best['clf']][0].fit(best['perm'][1], labels)

    # Store the Classifiers in Pickle
    # Pickle the following:
    #     1. The trained classifier
    #     2. The FE used (with its CV's and all) for re-extraction on pred data
    #     3. The feat_perm list of feats for re-extraction on prediction data
    clf_name = best['clf']
    to_pikle = (clfs[clf_name][0],
                clf_name,
                extractor.get_pickleables(),
                best['perm'][0])

    with open('sent.pkl', 'w') as f:
        pickle.dump(to_pikle, f)
