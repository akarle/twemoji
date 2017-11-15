"""
This requires that you've:
    1. pip install nltk
    2. Go to python command line
    3. import nltk
    4. nltk.download('wordnet')
    5. nltk.download('sentiwordnet')
"""
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
from CMUTweetTagger import runtagger_parse


def sentiwordnet_score(text):
    """
    Takes a tweet and returns a sentiment value based on sentiwordnet scores
    text: string of text (tweet)
    returns: float, positive values are positive sentiment,
                    negative values are negative sentiment
    """
    ws = [il[:2] for ol in runtagger_parse([text]) for il in ol]
    # print ws
    acceptable_pos = ['N', 'O', 'S', 'Z', 'V', 'A', 'R']
    wnl = WordNetLemmatizer()
    ws = [wnl.lemmatize(w[0].lower(), ark_to_swn(w[1])) +
          '.' + ark_to_swn(w[1]) + '.01'
          for w in ws if w[1] in acceptable_pos]
    # print ws
    score = 0.0
    for w in ws:
        try:
            s = swn.senti_synset(w)
            word_score = s.pos_score() - s.neg_score()
        except Exception:
            word_score = 0.0
        # print w, word_score
        score += word_score
    return score


def sentiwordnet_classify(text):
    """
    Returns a pos/neg/neutral sentiment val for text
        1 if pos
        0 if neutral
        -1 if neg
    """
    score = sentiwordnet_score(text)
    if score > 0:
        return 1
    elif score < 0:
        return -1
    else:
        return 0


def ark_to_swn(pos):
    if pos in ['N', 'O', '^', 'S', 'Z']:
        return 'n'
    elif pos is 'V':
        return 'v'
    elif pos is 'A':
        return 'a'
    elif pos is 'R':
        return 'r'
    else:
        return None
