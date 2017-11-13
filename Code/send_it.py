# Imports
from feature_extractor import FeatureExtractor
from feature_combinator import FeatureCombinator
from load_data import load_sent140
import os

# ##############################################
#               ARGUMENT PROCESSING
# ##############################################

# --> thinking sim args to run_me here...
# TODO
# examples
feats_to_extract = ['unigram', 'bigram']
clfs = ['nb', 'lr']

# ##############################################
#                   LOAD DATA
# ##############################################

data_path = os.path.join('..', 'Data', 'sent140', 'raw')
trdata, trlabels, tedata, telabels = load_sent140(data_path)
data = [trdata, trlabels, tedata, telabels]

# Randomize Data
# TODO

# ##############################################
#               EXTRACT FEATURES
# ##############################################

# Use FeatureExtractor (FE) to extract features
# FE will store the extracted features
fe = FeatureExtractor()
feats = fe.extract_features(['this is a test string',
                            'twemoji lets go baby'],
                            feats_to_extract)

# use FeatureCombnator to get all feat combinations
fc = FeatureCombinator(feats)

# ##############################################
#             INITIATE CLASSIFIERS
# ##############################################

# Initiate Classifiers to be trained


# ##############################################
#           TRAIN AND EVALUATE CLFS
# ##############################################

perm = fc.next_perm()
while perm is not None:
    print "Current perm: ", perm[0]
    print "Features Shape: ", perm[1].shape
    for clf in clfs:
        # Train (and Tune Hyperparams)

        # Score, save score

        # Add best clf to list of clfs for pickle
        pass

    perm = fc.next_perm()

# ##############################################
#               GRAPH EVALUATIONS
# ##############################################

# TODO -- use acc_graph already made

# ##############################################
#               SAVE TOP CLFS
# ##############################################

# Store the Classifiers in Pickle
# TODO
