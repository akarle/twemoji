# Imports

class FeatureExtractor():
	""" A class to extract text features """
	def __init__(self, arg):		
		self.arg = arg

	def unary_features(self, data):
		""" Extract Unary Features """
		pass

	def sent_analysis(self, data):
		""" Extracts Sentiment Analysis Features via sent_analysis.py """
		pass

	def extract_features(self, data, feat_types):
		""" External function to call to extract multiple features"""

		#thought: you'd pass [unary, binary] and extract and return both

