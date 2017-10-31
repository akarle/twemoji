# Imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os
import numpy as np
from scipy.sparse import hstack
from sent_analysis import SentimentClassifier

class FeatureExtractor():
    """ A class to extract text features """
    def __init__(self):
        # A mapping of keywords to functions (for use in extract features)
        self.features_dict = {'unary': self.unary_features, 'sent_analysis': self.sent_analysis}

    def unary_features(self, data):
        count_vect = CountVectorizer()
        return count_vect.fit_transform(data)
        
    def sent_analysis(self, data, sent_pred_func):
        """ Extracts Sentiment Analysis Features via sent_analysis.py """
        # Possible TODO: can move sent_clf to be an instance variable... I think it's cleaner
        # to have it isolated here, BUT if we end up calling sent_analysis mult times, then 
        # it will retrain sent_clf each time... (not an issue if this func used once)
        clf = MultinomialNB()
        sent_clf = SentimentClassifier(sent_pred_func, os.path.join('..', 'Data', 'sent140', 'raw'), clf)
        return sent_clf.fit_and_predict(data)

    def extract_features(self, data, features_to_extract):
        """ External function to call to extract multiple features 

            ie:
                features_to_extract = ['unary', ('sent_analysis', 'baseline')]
                would be 'unary' feats on data AND 'sent_analysis' of type 'baseline' on data
        """
        feats_not_initialized = True
        for feat in features_to_extract:
            print "Extracting feature", feat
            if type(feat) is tuple:
                #if its a tuple interpret it as a function name and args!
                if feats_not_initialized:
                    features = self.features_dict[feat[0]](data, *feat[1:])
                    feats_not_initialized = False
                else:
                    new_feats = self.features_dict[feat[0]](data, *feat[1:])
                    features = hstack((features, new_feats))

            else:
                if feats_not_initialized:
                    features = self.features_dict[feat](data)
                    feats_not_initialized = False
                else:
                    features = hstack((features, self.features_dict[feat](data)))

        return features
