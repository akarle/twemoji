"""
    This file is concerned with generating sentiment features,
    either classifications or probabilities, given a trained
    sentiment classifier and data to predict on.
"""
from text_feat_extractor import TextFeatureExtractor
from scipy.sparse import hstack


def predict_sent(data, clf, clf_name, pickleables, perm, prob=False):
    """ Given pickle stuff, return sent prediction on data """

    # Create new FE
    fe = TextFeatureExtractor()

    # Use FE w Pickleables to extract data on `data`
    feats_to_extract = list(perm)
    feats = fe.extract_features(data, feats_to_extract, pickles=pickleables)

    # Stack all feats (kinda like a 1-shot FC)
    feat_keys = feats.keys()
    full_feats = feats[feat_keys[0]]

    for k in feat_keys[1:]:
        full_feats = hstack((full_feats, feats[k]))

    # Use pretrained clf to predict on feats
    if not prob:
        preds = clf.predict(full_feats).reshape(-1, 1)
    else:
        preds = clf.predict_proba(full_feats)[:, -1].reshape(-1, 1)
        print "PROBS: ", preds[:10]

    # TODO: maybe adapt to be a dict of more than one clf?
    # FC is adaptable to this, would just need to pickle more than one!
    return {clf_name: preds}
