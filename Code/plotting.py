"""
    This is a file for all of our plotting needs, to keep code clean
    in other parts of the pipeline
"""

# Imports
from matplotlib import pyplot as plt
import itertools
import numpy as np


def acc_bar_chart(title, desc, baseline, acc_scores, labels, output_file, nofigs):
    """ Graph the given accuracies against each other and baseline """

    # Graph the scores
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.bar(range(1, len(acc_scores) + 1), acc_scores, tick_label=labels)
    plt.axhline(baseline, color='r', label="baseline")  # baseline
    plt.xticks(fontsize=8, rotation=45)

    # Give an appropriate bottom description
    # desc is a string of a list... make it a list
    descls = desc[1:-1].split(', ')
    descls = [x[2:-1] for x in descls]  # get rid of the unicode u' in string
    desc_str = "Preprocessing: " + ', '.join(descls)
    plt.subplots_adjust(bottom=.4)
    plt.text(.05, -.4, desc_str, fontsize=9, wrap=True)

    if not nofigs:
        plt.savefig(output_file)

    plt.clf()


def plot_confusion_matrix(cnf_mat, clf_name, feats_str, out_file, pipeline, nofigs):
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
    num_classes = cnf_mat.shape[0]

    if pipeline == 'emoji':
        title = "Emoji"
        classes = np.arange(num_classes)

    else:
        title = "Sentiment"
        classes = ["Negative", "Neutral", "Positive"]

    plt.imshow(cnf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()

    fmt = 'd'
    thresh = cnf_mat.max() / 2.
    for i, j in itertools.product(range(cnf_mat.shape[0]),
                                  range(cnf_mat.shape[1])):
        plt.text(j, i, format(cnf_mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_mat[i, j] > thresh else "black")

    ticks = np.arange(num_classes)
    plt.xticks(ticks, classes)
    plt.yticks(ticks, classes)


    plt.suptitle('%s %s Confusion Matrix' % (clf_name, title),
                 fontweight='bold')

    plt.title(feats_str)
    plt.ylabel('Gold Label')
    plt.xlabel('Predicted Label')

    if not nofigs:
        plt.savefig(out_file)

    plt.clf()
