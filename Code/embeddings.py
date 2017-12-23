"""
    This script is used to create the document embeddings used
    to train and evaluate models of the alternate pipeline
    (emb_run.py). It uses Doc2Vec to create an embedding
    for each tweet in the emoji dataset, and saves to file
"""
import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
import os
from load_data import load_emoji
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# load data
DATA_PATH = '../Data/noloc/tout.text'  # change this
LABEL_PATH = '../Data/noloc/lout.labels'  # change this
data, labels, count = load_emoji(DATA_PATH, LABEL_PATH)
clean_data = [d.lower() for d in data]

# Check if d2v already exists
if os.path.isfile('emoji.d2v'):
    print "loading d2v from file"
    model = Doc2Vec.load('emoji.d2v')

else:
    # create tagged documents for the doc2vec training
    print "training new d2v"
    docs = []
    for i in range(len(labels)):
        docs.append(TaggedDocument(simple_preprocess(clean_data[i]),
                    ['DATA_' + str(i)]))

    # initiate doc2vec model
    print "model initiated"
    model = Doc2Vec(size=100, workers=4)

    # build its vocab and train on the corpus -- note this is unsupervised
    model.build_vocab(docs)
    print "model vocab built"
    model.train(docs, total_examples=model.corpus_count, epochs=model.iter)
    print "model trained"

    # save the model for next time
    model.save('emoji.d2v')
    print "model saved"


# Now use model to create embedded sentences for the training data
embedded = [model.docvecs['DATA_' + str(i)] for i in range(len(labels))]

# Split data into train and test:
X_train, X_test, y_train, y_test = train_test_split(
    embedded, labels, test_size=0.30, random_state=0)
print "train/test split done"

# save to file for futher use
np.save('../Data/nolocDoc2Vec/%s.npy' % 'xtrain', X_train)
np.save('../Data/nolocDoc2Vec/%s.npy' % 'xtest', X_test)
np.save('../Data/nolocDoc2Vec/%s.npy' % 'ytrain',  y_train)
np.save('../Data/nolocDoc2Vec/%s.npy' % 'ytest', y_test)

# basic sklearn classifiaction!
lr = LogisticRegression()
lr.fit(X_train, y_train)
print "fit LR"
print lr.score(X_test, y_test)
