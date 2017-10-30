# Imports
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

from load_data import load_sent140

class SentimentClassifier():
    def __init__(self, data_path, model):
        self.trdata = None
        self.tedata = None
        self.trlabels = None
        self.telabels = None
        self.data_path = data_path
        self.model = model

    def load_data(self):
        """ A function to load data for the sentiment classifier 
        
            The thought here is that separating this from the constructor prevents lag
            when instantiating (also maybe future capability of limiting how much data to load!)
            TODO: limit amount to load
        """
        self.trdata, self.trlabels, self.tedata, self.telabels = load_sent140(self.data_path)

    def fit_and_predict_baseline(self, data):
        """ Fit for baseline sentiment scores and predict on incoming data """
        # Fit to training data
        count_vect = CountVectorizer()
        if self.trdata is None:
            self.load_data()

        x_counts = count_vect.fit_transform(self.tedata)
        self.model.fit(x_counts, self.telabels)
        print 'Model fit'
        # Use fit model to predict on data:
        # transform data
        counts = count_vect.transform(data)
        return self.model.predict(counts)