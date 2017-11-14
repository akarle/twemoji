# Imports
from text_feat_extractor import TextFeatureExtractor
from feature_combinator import FeatureCombinator
from sklearn.metrics import accuracy_score
import argparse
from baseline import baseline_predict
from load_data import load_sent140
import os

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
clfs = ['nb', 'lr']

# ##############################################
#                   LOAD DATA
# ##############################################

data_path = os.path.join('..', 'Data', 'sent140', 'raw')

if args.num_instances:
    X_train, y_train, X_test, y_test = load_sent140(data_path,
                                                    args.num_instances[0])
else:
    X_train, y_train, X_test, y_test = load_sent140(data_path)

# Randomize Data
# TODO

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


# ##############################################
#           TRAIN AND EVALUATE CLFS
# ##############################################

feat_perm = fc.next_perm()
while feat_perm is not None:
    print "Current perm: ", feat_perm[0]
    print "Features Shape: ", feat_perm[1].shape
    for clf in clfs:
        # Train (and Tune Hyperparams)

        # Score, save score

        # Add best clf to list of clfs for pickle
        pass

    feat_perm = fc.next_perm()


# Baseline score
print "*******"
print "Calculating baseline..."
baseline_score = accuracy_score(y_test, baseline_predict(y_test))
print "Baseline accuracy score: ", baseline_score
print "*******"

# ##############################################
#               GRAPH EVALUATIONS
# ##############################################

# TODO -- use acc_graph already made

# ##############################################
#               SAVE TOP CLFS
# ##############################################

# Store the Classifiers in Pickle
# TODO
