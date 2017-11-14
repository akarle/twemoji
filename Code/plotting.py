# Imports
from matplotlib import pyplot as plt


def acc_bar_chart(title, desc, baseline, acc_scores, labels, output_file):
    """ Graph the given accuracies against each other and baseline """
    plt.title(title)
    plt.bar(range(1, len(acc_scores) + 1), acc_scores, tick_label=labels)
    plt.axhline(baseline, color='r', label="baseline")
    plt.savefig(output_file)
    plt.clf()
