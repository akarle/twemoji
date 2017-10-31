""" File to load in the data

    Loads the tweets into a list of strings `data`
    Loads the labels into a seperate list, `labels`
"""
import csv
import os


def load_data(data_file_path, label_file_path, num_instances=float('inf')):
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


def load_sent140(data_path):
    """ A func to load in the sentiment140 dataset

        data_path leads to a dir with contents test.csv and train.csv
    """
    # Helper func (as same thing done to both test and train)
    def parse_file(file_path):
        print 'Loading %s' % file_path
        labels = []
        text = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter=',', quotechar='\"')
            for l in reader:
                labels.append(l[0])
                text.append(l[-1].rstrip())
                if len(l) != 6:
                    print "CSV PARSE ERROR"

        labels = map(int, labels)
        return text, labels

    train_file = os.path.join(data_path, 'train.csv')
    test_file = os.path.join(data_path, 'test.csv')

    trdata, trlabels = parse_file(train_file)
    tedata, telabels = parse_file(test_file)

    return (trdata, trlabels, tedata, telabels)
