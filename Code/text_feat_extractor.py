# Imports
from sklearn.feature_extraction.text import CountVectorizer
import CMUTweetTagger


class TextFeatureExtractor():
    """ A class to extract text features """
    def __init__(self):
        # A mapping of keywords to functions (for use in extract features)
        self.features_dict = {'unigram': self.unigram_features,
                              'bigram': self.bigram_features}

    def unigram_features(self, data, cvargs):
        if cvargs is not None:
            count_vect = CountVectorizer(
                lowercase=cvargs['lowercase'],
                stop_words=cvargs['stop_words'],
                strip_accents=cvargs['strip_accents'],
                token_pattern=cvargs['token_pattern']
            )
        else:
            count_vect = CountVectorizer()
        return count_vect.fit_transform(data)

    def bigram_features(self, data, cvargs):
        if cvargs is not None:
            count_vect = CountVectorizer(
                ngram_range=(2, 2),
                lowercase=cvargs['lowercase'],
                stop_words=cvargs['stop_words'],
                strip_accents=cvargs['strip_accents'],
                token_pattern=cvargs['token_pattern']
            )
        else:
            count_vect = CountVectorizer(ngram_range=(2, 2))
        return count_vect.fit_transform(data)

    def extract_features(self, data, features_to_extract, cvargs=None):
        """ External function to call to extract multiple features

            ie:
                features_to_extract = ['unigram', 'bigram']
                would extract both feats and save to FE
        """

        features = {}
        for feat in features_to_extract:
            print "Extracting feature", feat
            features[feat] = self.features_dict[feat](data, cvargs)

        return features
