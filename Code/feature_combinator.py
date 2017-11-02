from itertools import combinations
from scipy.sparse import hstack


class FeatureCombinator():
    """ FeatureCombinator is a class to get all combos of feats

        More specifically, given `text_feats` and `clf_preds` it will get
        all permutations of all combos of text feats w one or zero clf preds.

        It does not store actual feature vectors (due to memory cost), but
        instead stores a representation of them, and one can iterate through
        them using `next_perm`
    """
    def __init__(self, feats, preds={}):
        self.feats = feats  # A dict of {'feat_name': <feat vector>}
        self.clf_preds = preds  # A dict of {'clf_name': <predicted feats>}
        self.feat_perms = self.get_all_perms()  # all permutations of features
        self.curr_perm = 0  # the permutation to be returned

    def get_all_perms(self):
        """ Returns a list of all permutations of text feats and clf preds

            WANT: a list of name perms, NOT the actual data (too big)
        """
        feat_names = self.feats.keys()

        # Get all combos of text features
        combs = [comb for i in range(len(feat_names))
                 for comb in combinations(feat_names, i + 1)]

        # Instantiate combined list to return (with clf_preds)
        pred_combs = combs[:]  # slice to copy

        # Add a copy of combs with each prediction appended
        for pred in self.clf_preds.keys():
            pred_combs += [(pred,)]
            pred_combs += [combs[i] + (pred,) for i in range(len(combs))]

        print "Combos of feats given: ", pred_combs
        return pred_combs

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
