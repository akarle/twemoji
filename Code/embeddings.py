from gensim.models import Word2Vec
import numpy as np
from load_data import load_emoji


DATA_PATH = '../Data/noloc/tout.text'  # change this
LABEL_PATH = '../Data/noloc/lout.labels'  # change this
data, labels, count = load_emoji(DATA_PATH, LABEL_PATH)
clean_data = [d.lower().split() for d in data]
model = Word2Vec(clean_data, size=100, window=5, min_count=5, workers=4)
# Save Word2Vec
# model.save('../Data/word2vec-embeddings')
# Save numpy
np.save('../Data/np-w2v-embeddings', model.wv.syn0)
