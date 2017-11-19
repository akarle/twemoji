""" File to load in the data

    Loads the tweets into a list of strings `data`
    Loads the labels into a seperate list, `labels`
"""
import csv
import os
from collections import defaultdict


def load_emoji(data_file_path, label_file_path, num_instances=float('inf')):
    """ A function to load in the semeval data

        Example params:
            label_file = os.path.join('..','Data','trial', 'us_trial.labels')
            data_file = os.path.join('..','Data','trial', 'us_trial.text')

    """
    # load in data
    tf = open(data_file_path, 'r')
    lf = open(label_file_path, 'r')

    data = []
    labels = []

    count = 0
    for tweet in tf:
        if count >= num_instances:
            break
        label = lf.readline().rstrip()  # rstrip to remove trailing \n
        data.append(tweet.rstrip())
        labels.append(label)
        count += 1

    tf.close()
    lf.close()

    # convert the labels to ints
    labels = map(int, labels)

    return (data, labels, count)


def load_sent140(data_path, dataset='test', num_instances=float('inf')):
    """ A func to load in the sentiment140 dataset

        data_path leads to a dir with a csv file of data
        @param `dataset` is the name of the csv file
    """

    file_path = os.path.join(data_path, dataset + '.csv')
    print 'Loading %s' % file_path
    labels = []
    data = []
    with open(file_path, 'r') as f:
        unicode_err_count = 0
        word_count = 0
        reader = csv.reader(f, delimiter=',', quotechar='\"')
        for l in reader:
            if word_count >= num_instances:
                break

            word = l[0].rstrip()

            # Only extract strings without unicode errors!
            try:
                word.decode('utf-8')  # attempt a decode
                data.append(word)  # append if success
                labels.append(l[-1])
                word_count += 1

            except UnicodeDecodeError:
                unicode_err_count += 1

    labels = map(int, labels)
    labels = map(lambda x: x / 2, labels)  # map 2->1, 4->2
    print 'Unicode Error Count: ', unicode_err_count

    file_path = os.path.join(data_path, dataset + '.csv')

    return (data, labels, word_count)


def load_nrc_emotion_lexicon(
    data_path='../Data/Lexicons/' +
        'NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'):
    emolex = {}
    emolex['anger'] = defaultdict(int)
    emolex['anticipation'] = defaultdict(int)
    emolex['disgust'] = defaultdict(int)
    emolex['fear'] = defaultdict(int)
    emolex['joy'] = defaultdict(int)
    emolex['positive'] = defaultdict(int)
    emolex['negative'] = defaultdict(int)
    emolex['sadness'] = defaultdict(int)
    emolex['surprise'] = defaultdict(int)
    emolex['trust'] = defaultdict(int)
    with open(data_path, 'rb') as fp:
        reader = csv.reader(fp, delimiter='\t')
        data = []
        for row in reader:
            data.append(row)
        for d in data:
            if len(d) == 3:
                emolex[str(d[1])][str(d[0])] = int(d[2])
    # for d in emolex:
    #     print emolex[d]['phony']
    return emolex
