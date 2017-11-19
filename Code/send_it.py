# Imports
import random
from text_feat_extractor import TextFeatureExtractor
from feature_combinator import FeatureCombinator
from sklearn.metrics import accuracy_score
import argparse
from baseline import baseline_predict
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from load_data import load_sent140
from sklearn.model_selection import train_test_split
import os
import pickle
from sklearn.externals import joblib

# ##############################################
#               ARGUMENT PROCESSING
# ##############################################

# --> thinking sim args to run_me here...
# TODO
parser = argparse.ArgumentParser(
    description="Train sentiment classifiers output results.")

parser.add_argument('-n',
                    nargs=1,
                    type=int,
                    help='number of data entries to train/evaluate on',
                    metavar='N',
                    dest='num_instances')

args = parser.parse_args()

# examples
feats_to_extract = ['unigram', 'bigram']

# ##############################################
#                   LOAD DATA
# ##############################################

data_path = os.path.join('..', 'Data', 'sent140')

if args.num_instances:
    X_train, y_train, _, _, _, _ = load_sent140(data_path, args.num_instances[0])
else:
    X_train, y_train, _, _, _, _ = load_sent140(data_path)

print (len(X_train), len(y_train))
for i in range(10):
    print (X_train[i], y_train[i])

# Randomize data order to prevent overfitting to subset of
# data when running on fewer instances
combined = list(zip(X_train, y_train))
random.shuffle(combined)
X_train[:], y_train[:] = zip(*combined)


# ##############################################
#               EXTRACT FEATURES
# ##############################################

# Use FeatureExtractor (FE) to extract features
# FE will store the extracted features
fe = TextFeatureExtractor()
feats = fe.extract_features(X_train, feats_to_extract)

# use FeatureCombnator to get all feat combinations
fc = FeatureCombinator(feats)

# ##############################################
#             INITIATE CLASSIFIERS
# ##############################################

# Initiate Classifiers to be trained
nb = MultinomialNB()
lr = LogisticRegression()

clfs = {'nb': nb, 'lr': lr}

# ##############################################
#           TRAIN AND EVALUATE CLFS
# ##############################################
scores = {}
best = {'clf': None, 'perm': None, 'score': 0}

feat_perm = fc.next_perm()
while feat_perm is not None:
    print "Current perm: ", feat_perm[0]
    print "Features Shape: ", feat_perm[1].shape

    # Split data into train and test:
    curr_X_tr, curr_X_te, curr_y_tr, curr_y_te = \
        train_test_split(feat_perm[1], y_train, test_size=0.30, random_state=0)

    for c in clfs:
        # Train (and Tune Hyperparams)
        clfs[c].fit(curr_X_tr, curr_y_tr)

        # Score, save score
        preds = clfs[c].predict(curr_X_te)
        score = accuracy_score(curr_y_te, preds)
        score_key = c + str(feat_perm[0])
        scores[score_key] = score

        print "Average accuracy score for %s with feats %s: %f" \
              % (c, feat_perm[0], score)

        # Add best clf to list of clfs for pickle
        if score > best['score']:
            print "Found new best!"
            best['score'] = score
            best['clf'] = c
            best['perm'] = feat_perm

    feat_perm = fc.next_perm()

# Baseline score
print "*******"
print "Calculating baseline..."
mc_label, baseline_preds = baseline_predict(curr_y_te)
baseline_score = accuracy_score(curr_y_te, baseline_preds)
print "Most common label: ", mc_label
print "Baseline accuracy score: ", baseline_score
print "*******"

# ##############################################
#               GRAPH EVALUATIONS
# ##############################################

# TODO -- use acc_graph already made

# ##############################################
#               SAVE TOP CLFS
# ##############################################

# Pull best from evaluation from curr_best
print "Best overall: %s with %s giving score %f" % \
      (best['clf'], best['perm'][0], best['score'])

# Retrain the model with the data
clfs[best['clf']].fit(best['perm'][1], y_train)

# Store the Classifiers in Pickle
# Pickle the following:
#     1. The trained classifier
#     2. The FE used (with its CV's and all) for re-extraction on pred data
#     3. The feat_perm list of feats for re-extraction on prediction data
clf_name = best['clf']
to_pikle = (clfs[clf_name], clf_name, fe.get_pickleables(), best['perm'][0])

with open('sent.pkl', 'w') as f:
    pickle.dump(to_pikle, f)
