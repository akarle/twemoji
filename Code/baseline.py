import numpy as np


def baseline_predict(labels):
    """ A function that predicts the most common label """
    # get the most common label:
    labels_array = np.array(labels)
    mc_label = np.argmax(np.bincount(labels_array))

    # return an array of the most common label (one per data case)
    preds = np.multiply(mc_label, np.ones(len(labels)))
    # print preds
    return (mc_label, preds)
