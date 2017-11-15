# Imports
from matplotlib import pyplot as plt
import itertools
import numpy as np


def acc_bar_chart(title, desc, baseline, acc_scores, labels, output_file):
    """ Graph the given accuracies against each other and baseline """

    # Graph the scores
    plt.title(title)
    plt.bar(range(1, len(acc_scores) + 1), acc_scores, tick_label=labels)
    plt.axhline(baseline, color='r', label="baseline")  # baseline

    # Give an appropriate bottom description
    # desc is a string of a list... make it a list
    descls = desc[1:-1].split(', ')
    descls = [x[2:-1] for x in descls]  # get rid of the unicode u' in string
    desc_str = "Preprocessing: " + ', '.join(descls)
    plt.subplots_adjust(bottom=.15)
    plt.text(.05, -.04, desc_str, fontsize=9, wrap=True)
    plt.savefig(output_file)
    plt.clf()


def plot_confusion_matrix(cnf_mat, clf_name, feats_str, output_file):
    """ Creates and plots a confusion matrix for the predictions

        CITATION: THIS CODE HEAVILY ADAPTED FROM SCIKITLEARN:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    # Create Mapping to Emojis for Display TODO
    # us_map_file = os.path.join('..', 'Data', 'mappings', 'us_mapping.txt')
    # emoji_path = os.path.join('..', 'Data', 'mappings', 'emojis')
    # us_map = {}
    # with open(us_map_file, 'r') as f:
    #     for line in f:
    #         toks = line.split()
    #         em_file = os.path.join(emoji_path, toks[2][1:-1] + '.png')
    #         us_map[int(toks[0])] = plt.imread(em_file)

    # Map the preds and gold to the emojis (for display)
    # TODO
    classes = range(10)
    n_clss = len(classes)

    plt.imshow(cnf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()

    fmt = 'd'
    thresh = cnf_mat.max() / 2.
    for i, j in itertools.product(range(cnf_mat.shape[0]),
                                  range(cnf_mat.shape[1])):
        plt.text(j, i, format(cnf_mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_mat[i, j] > thresh else "black")

    ticks = np.arange(n_clss)
    plt.xticks(ticks, classes)
    plt.yticks(ticks, classes)

    plt.suptitle('%s Emoji Confusion Matrix' % clf_name, fontweight='bold')
    plt.title(feats_str)
    plt.ylabel('Gold Label')
    plt.xlabel('Predicted Label')

    plt.savefig(output_file)
    plt.clf()
