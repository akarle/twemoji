# Imports
from sklearn.feature_extraction.text import CountVectorizer
import CMUTweetTagger


class TextFeatureExtractor():
    """ A class to extract text features """
    def __init__(self):
        # A mapping of keywords to functions (for use in extract features)
        self.features_dict = {'unigram': self.unigram_features,
                              'bigram': self.bigram_features}

    def unigram_features(self, data, analyzer):
        a=lambda x: [[il[:2] for il in ol] for ol in CMUTweetTagger.runtagger_parse([x])]
        flatten = lambda l: [item for sublist in l for item in sublist]
        count_vect = CountVectorizer(tokenizer=lambda x: flatten(a(x)), strip_accents='unicode')
        return count_vect.fit_transform(data)

    def bigram_features(self, data, analyzer):
        a=lambda x: [[il[:2] for il in ol] for ol in CMUTweetTagger.runtagger_parse([x])]
        flatten = lambda l: [item for sublist in l for item in sublist]
        count_vect = CountVectorizer(ngram_range=(2, 2), tokenizer=lambda x: flatten(a(x)), strip_accents='unicode')
        return count_vect.fit_transform(data)

    def extract_features(self, data, features_to_extract, analyzer='word'):
        """ External function to call to extract multiple features

            ie:
                features_to_extract = ['unigram', 'bigram']
                would extract both feats and save to FE
        """

        if analyzer is None:
            analyzer = 'word'
        features = {}
        for feat in features_to_extract:
            print "Extracting feature", feat
            features[feat] = self.features_dict[feat](data, analyzer)

        return features
