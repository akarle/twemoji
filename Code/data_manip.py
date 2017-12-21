""" A file to manipulate raw data as needed """
import os
import csv
import random

# Sentiment 140:
# 1. Strip unnecessary data (only tweets / labels)
# 2. Shuffle data (so as a partial read has more than 1 label)


def manip_file(file_path, output_path,
               tweetind, labelind, delimiter=','):
    labels = []
    text = []
    label_map = {'neutral': 2, 'positive': 4, 'negative': 0}

    # Read in the raw data, save only parts we care about
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar='\"')
        for l in reader:
            label = l[labelind]
            if type(label) is str and not label.isdigit():
                label = label_map[label]

            labels.append(label)
            text.append(l[tweetind].rstrip())

    # Shuffle the lists
    combined = list(zip(text, labels))
    random.shuffle(combined)
    text, labels = zip(*combined)

    # Write the shuffled out to file
    with open(output_path, 'w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='\"')
        for i in range(len(text)):
            writer.writerow([text[i], labels[i]])


if __name__ == "__main__":
    data_path = os.path.join('..', 'Data', 'sent140', 'raw')
    output_path = os.path.join('..', 'Data', 'sent140')

    train_file = os.path.join(data_path, 'train.csv')
    test_file = os.path.join(data_path, 'test.csv')

    train_out = os.path.join(output_path, 'train.csv')
    test_out = os.path.join(output_path, 'test.csv')

    manip_file(train_file, train_out, -1, 0)
    manip_file(test_file, test_out, -1, 0)

    # The semeval data
    semeval_raw = os.path.join('..', 'Data', 'semeval', 'raw', 'train.csv')
    semeval_out = os.path.join('..', 'Data', 'semeval', 'train.csv')

    manip_file(semeval_raw, semeval_out, 2, 1, delimiter='\t')
