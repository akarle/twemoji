""" File to load in the data

    Loads the tweets into a list of strings `data`
    Loads the labels into a seperate list, `labels`
"""

def load_data(data_file_path, label_file_path):
    """ A function to load in the data

        Example params:
            label_file = os.path.join('..','Data','trial', 'us_trial.labels')
            data_file = os.path.join('..','Data','trial', 'us_trial.text')

    """
    # load in data
    tf = open(data_file_path, 'r')
    lf = open(label_file_path, 'r')

    data = []
    labels = []


    for tweet in tf:
        label = lf.readline().rstrip() #rstrip to remove trailing \n
        data.append(tweet.rstrip())
        labels.append(label)

    tf.close()
    lf.close()

    #convert the labels to ints
    labels = map(int, labels)

    print 'First 10 tweets: ', data[:10]
    print 'First 10 labels: ', labels[:10]

    return (data, labels)