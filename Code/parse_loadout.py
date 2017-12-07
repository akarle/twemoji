import json
import os

"""
For use in JSON loadout files:

Classifiers:
    nb: Multinomial Naive Bayes
    lr: LogisticRegression
Preprocessing:
    lowercase: convert all words to lowercase
    remove-common-words: removes stop words from list
    strip-accents: converts characters with accents to regular characters
    count-singles: count single characters as tokens,
                   otherwise tokens have length mininum 2
    spell-correction: autocorrect spelling mistakes
    pos-tags: add Ark Tweet POS tags to each word
    ----- The following require pos-tags ------
    strip-at-mentions: don't count @user mentions as tokens
    strip-punctuation: don't count punctionation as tokens (default w/o POS)
    strip-hashtags: don't count hashtags as tokens
    strip-discourse-markers: don't count discourse markers as tokens
    strip-urls: don't count urls as tokens
    -------------------------------------------
Text Features:
    unigrams
    bigrams
"""


def parse_loadout(loadout_name):
    """
    Returns 5-tuple:
        classifiers: list
        manual preprocessing flags: list
        CountVectorizer args: dict
        features: list
        preprocessing description for plots: string
    """
    loadout_path = os.path.join("..", "Loadouts", loadout_name + ".json")
    with open(loadout_path) as file:
        loadout = json.load(file)

        # Classifiers
        classifiers = [c.encode('ascii') for c in loadout['classifiers']]

        # Manual preprocessing flags
        manpre = []
        if 'spell-correction' in loadout['preprocessing']:
            manpre.append('spell-correction')
        if 'word-clustering' in loadout['preprocessing']:
            manpre.append('word-clustering')
        if 'word-clustering-append' in loadout['preprocessing']:
            manpre.append('word-clustering-append')
        if 'pos-tags' in loadout['preprocessing']:
            manpre.append('pos-tags')

        # CountVectorizer args
        cvargs = {}
        cvargs['lowercase'] = 'lowercase' in loadout['preprocessing']
        if 'remove-common-words' in loadout['preprocessing']:
            swfile = open(os.path.join('..', 'Loadouts', 'stopwords.txt'), 'r')
            cvargs['stop_words'] = [l.rstrip() for l in swfile.readlines()]
        else:
            cvargs['stop_words'] = None
        if 'strip-accents' in loadout['preprocessing']:
            cvargs['strip_accents'] = 'unicode'
        else:
            cvargs['strip_accents'] = None
        cvargs['token_pattern'] = build_regex(
            'strip-punctuation' in loadout['preprocessing'],
            'count-singles' in loadout['preprocessing'],
            'pos-tags' in loadout['preprocessing'],
            'strip-hashtags' in loadout['preprocessing'],
            'strip-at-mentions' in loadout['preprocessing'],
            'strip-discourse-markers' in loadout['preprocessing'],
            'strip-urls' in loadout['preprocessing']
        )

        # Features
        features = [f.encode('ascii') for f in loadout['text-features']]

        return(
            classifiers,
            manpre,
            cvargs,
            features,
            str(loadout['preprocessing'])
        )


def build_regex(punc, single, pos, hashtag, atm, dcm, url):
    # r'(?u)\bpos_._\w+\b'
    regex = r'(?u)\b'
    if pos:
        regex += r'[pP][oO][sS]_[NnOo^SsZzVvAaRr!DdPp&TtXx$LlMmYyEe'
        if not punc:
            regex += r',Gg'
        if not hashtag:
            regex += r'#'
        if not atm:
            regex += r'@'
        if not dcm:
            regex += r'~'
        if not url:
            regex += r'Uu'
        regex += r']_\S'
        if not single:
            regex += r'\S'
        regex += r'+\b'
    else:
        regex += r'\w'
        if not single:
            regex += r'\w'
        regex += r'+\b'
    return regex
