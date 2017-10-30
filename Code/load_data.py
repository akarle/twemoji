""" File to load in the data

    Loads the tweets into a list of strings `data`
    Loads the labels into a seperate list, `labels`
"""

def load_data(data_file_path, label_file_path, num_instances=float('inf'), verbose=1):
    """ A function to load in the data

        Example params:
            label_file = os.path.join('..','Data','trial', 'us_trial.labels')
            data_file = os.path.join('..','Data','trial', 'us_trial.text')

    """
    if verbose >= 1:
        def verboseprint(*args):
            for arg in args:
               print arg,
            print
    else:
        verboseprint = lambda *a: None

    # load in data
    tf = open(data_file_path, 'r')
    lf = open(label_file_path, 'r')

    data = []
    labels = []

    count = 0
    for tweet in tf:
        if count >= num_instances:
            break
        label = lf.readline().rstrip() #rstrip to remove trailing \n
        data.append(tweet.rstrip())
        labels.append(label)
        count += 1

    tf.close()
    lf.close()

    #convert the labels to ints
    labels = map(int, labels)

    verboseprint("Loaded ", count, " tweets...")
    verboseprint('First 10 tweets and labels: ')
    verboseprint("|   Label ::: Tweet")
    verboseprint("|   ---------------")
    for i in range (10):
        verboseprint('|%6s' % labels[i], " ::: ", data[i])
    verboseprint("*******")

    return (data, labels)
