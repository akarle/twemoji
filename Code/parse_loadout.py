import json
import os
from sklearn.feature_extraction.text import CountVectorizer


"""
For use in JSON loadout files:

Classifiers:
    nb: Multinomial Naive Bayes
    lr: LogisticRegression
Preprocessing:
    lowercase: convert all words to lowercase
    remove-common-words: removes stop words from list
    strip-accents: converts characters with accents to regular characters
    strip-punctuation: don't count punctionation in tokens
    count-singles: count single characters as tokens,
                   otherwise tokens have length mininum 2
    strip-at-mentions: don't count @user mentions as tokens
    TODO: spell-correction
    TODO: remove-location
    TODO: pos-tags
Text Features:
    unigrams
    bigrams
"""


def parse_loadout(loadout_name):
    """
    Returns 4-tuple:
        classifiers: list
        preprocessing: list
        analyzer: analyzer
        features: list
    """
    loadout_path = os.path.join("..", "Loadouts", loadout_name + ".json")
    with open(loadout_path) as file:
        loadout = json.load(file)
        # Build analyzer for preprocessing
        lowercase = 'lowercase' in loadout['preprocessing']
        if 'remove-common-words' in loadout['preprocessing']:
            swfile = open(os.path.join('..', 'Loadouts', 'stopwords.txt'), 'r')
            stop_words = [line.rstrip() for line in swfile.readlines()]
        else:
            stop_words = None
        if 'strip-at-mentions' in loadout['preprocessing']:
            if stop_words is None:
                stop_words = []
                stop_words.append('user')
        if 'strip-accents' in loadout['preprocessing']:
            strip_accents = 'unicode'
        else:
            strip_accents = None
        strip_punctation = 'strip-punctuation' in loadout['preprocessing']
        count_singles = 'count-singles' in loadout['preprocessing']
        cv = CountVectorizer(strip_accents=strip_accents, lowercase=lowercase,
                             stop_words=stop_words, token_pattern=build_regex(
                                 strip_punctation,
                                 count_singles,
                             ))
        # Manual preprocessing
        manpre = []
        if 'spell-correction' in loadout['preprocessing']:
            manpre.append('spell-correction')
        if 'remove-location' in loadout['preprocessing']:
            manpre.append('remove-location')
        if 'pos-tags' in loadout['preprocessing']:
            manpre.append('pos-tags')
        return(
            [c.encode('ascii') for c in loadout['classifiers']],
            manpre,
            cv.build_analyzer(),
            [f.encode('ascii') for f in loadout['text-features']]
        )


def build_regex(punc, single):
    regex = ""
    regex += "['a-zA-Z0-9"
    if not punc:
        regex += "-!$%^&#*()_+|~=`{}\[\]:\";<>?,.\/"
    regex += "]"
    if not single:
        regex += regex
    regex += "+"
    return regex
