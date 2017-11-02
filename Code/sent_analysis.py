""" DEPRECATED ON THE NEWEST FEATURE COMBINATOR BRANCH!

    TO BE DELETED. SEE PULL REQUEST #7
"""


# # Imports
# # from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import CountVectorizer

# from load_data import load_sent140


# class SentimentClassifier():
#     def __init__(self, pred_func, data_path, model):
#         # Data related vars
#         self.trdata = None
#         self.tedata = None
#         self.trlabels = None
#         self.telabels = None
#         self.data_path = data_path

#         # Prediction related features
#         self.model = model

#         # Use for a mapping of what prediction func to use!
#         self.pred_dict = {'baseline': self.fit_and_predict_baseline}
#         self.pred_func_name = pred_func

#     def fit_and_predict(self, data):
#         """ Prediction wrapper that uses self.type_dict and self.type
#         to call the right prediction function """
#         print "Using SentimentClassifier to predict sentiment using the \
#                 function ", self.pred_func_name

#         return self.pred_dict[self.pred_func_name](data)

#     def load_data(self):
#         """
#             A function to load data for the sentiment classifier

#             The thought here is that separating this from the constructor
#             prevents lag when instantiating (also maybe future capability
#             of limiting how much data to load!)
#             TODO: limit amount to load
#         """
#         self.trdata, self.trlabels, self.tedata, \
#             self.telabels = load_sent140(self.data_path)

#     def fit_and_predict_baseline(self, data):
#         """ Fit for baseline sentiment scores and predict on incoming data
#             DON'T CALL DIRECTLY, USE `fit_and_predict`
#         """
#         # Fit to training data
#         count_vect = CountVectorizer()
#         if self.trdata is None:
#             self.load_data()

#         x_counts = count_vect.fit_transform(self.tedata)
#         self.model.fit(x_counts, self.telabels)
#         print 'Model fit'
#         # Use fit model to predict on data:
#         # transform data
#         counts = count_vect.transform(data)
#         preds = self.model.predict(counts)  # numpy array
#         return preds.reshape(len(preds), 1)  # numpy nx1 column
