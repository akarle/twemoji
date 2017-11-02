from itertools import combinations
from scipy.sparse import hstack


class FeatureCombinator():
    def __init__(self, feats, data=[], sent_clfs={}):
        self.feats = feats  # A dict of {'feat_name': <feat vector>}
        self.sent_clfs = sent_clfs  # best sent classifiers from pickle
        self.feat_perms = self.get_all_perms()  # all permutations of features
        self.curr_perm = 0  # the permutation to be returned
        self.clf_preds = self.preds_from_clfs(data)  # cached predictions

    def get_all_perms(self):
        """ Returns a list of all permutations of text feats and clf preds

            WANT: a list of name perms, NOT the actual data (too big)
        """
        feat_names = self.feats.keys()

        # TODO: COMBINE EACH COMB OF FEAT WITH 0 OR 1 CLF
        # clf_names = self.sent_clfs.keys()
        # combined_names = feat_names + clf_names

        combs = [comb for i in range(len(feat_names))
                 for comb in combinations(feat_names, i + 1)]

        print "Combos of feats given: ", combs
        return combs

    def preds_from_clfs(self, data):
        """ Cache the preds on the data for use in perm combos """
        if len(data) == 0:
            return None
        else:
            pass  # TODO

    def next_perm(self):
        """ Use feat extractor and clfs to get, combine, return a perm """
        if self.curr_perm < len(self.feat_perms):
            perm = self.feat_perms[self.curr_perm]
            self.curr_perm += 1

            # Go through perm and combine the feats
            # ----> use hstack from scipy
            # ----> if classifier, get preds (TODO: cache them in self?)

            # perm is a tuple of feats to combine
            features = self.feats[perm[0]]
            for feat in perm[1:]:
                features = hstack((features, self.feats[feat]))

            return (perm, features)

        else:
            return None
