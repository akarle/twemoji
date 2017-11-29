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
import os
from load_data import load_nrc_emotion_lexicon as load_emolex
import csv
import unicodedata
import sys
import random
import re
import numpy as np


def sentiwordnet_score(text):
    """
    Takes a tweet and returns a sentiment value based on sentiwordnet scores
    text: string of text (tweet)
    returns: float, positive values are positive sentiment,
                    negative values are negative sentiment
    """
    nfkd_form = unicodedata.normalize('NFKD', unicode(text, 'utf8'))
    utext = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
    ws = [il[:2] for ol in runtagger_parse([utext]) for il in ol]
    # print ws
    acceptable_pos = ['N', 'O', 'S', 'Z', 'V', 'A', 'R']
    wnl = WordNetLemmatizer()
    ws = [wnl.lemmatize(w[0].lower()) +
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


def emolex_score(tlist):
    """
    tlist: a list of strings
    """
    emolex = load_emolex()
    wnl = WordNetLemmatizer()
    feats = None
    for t in tlist:
        tfeat = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for word in re.findall(r'(?u)\b\w+\b', t.lower()):
            # print word
            word = wnl.lemmatize(word)
            tfeat[0] += emolex['anger'][word]
            tfeat[1] += emolex['anticipation'][word]
            tfeat[2] += emolex['disgust'][word]
            tfeat[3] += emolex['fear'][word]
            tfeat[4] += emolex['joy'][word]
            tfeat[5] += emolex['positive'][word]
            tfeat[6] += emolex['negative'][word]
            tfeat[7] += emolex['sadness'][word]
            tfeat[8] += emolex['surprise'][word]
            tfeat[9] += emolex['trust'][word]
        if feats is not None:
            feats = np.append(feats, [tfeat], axis=0)
        else:
            feats = np.array([tfeat])
    return feats


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


def test_classifier(numinstances):
    # data_path = os.path.join('..', 'Data', 'sent140')
    # _, _, test, labels = load_sent140(data_path)
    with open('../Data/sad.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = []
        for row in reader:
            data.append(row)
        random.shuffle(data)
        pcorrect = 0.0
        ncorrect = 0.0
        pwrong = 0.0
        nwrong = 0.0
        neutral = 0
        count = 0
        for row in data:
            if row[1] == 'Sentiment':
                continue
            gold = int(row[1])
            if count % 10 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            if count == numinstances:
                break
            try:
                sm = emolex_score([row[3]])[0]
                # print sm
                pos_pre = sm[5]
                neg_pre = sm[6]
                if pos_pre == neg_pre:
                    if gold == 0:
                        neutral -= 1
                    else:
                        neutral += 1
                    continue
                predict = pos_pre >= neg_pre
            except Exception:
                continue
            if predict:
                if gold is 1:
                    pcorrect += 1
                else:
                    nwrong += 1
            else:
                if gold is 1:
                    pwrong += 1
                else:
                    ncorrect += 1
            count += 1
        acc = (pcorrect + ncorrect) / (pcorrect + ncorrect + pwrong + nwrong)
        pre = pcorrect / (pcorrect + nwrong)
        rec = pcorrect / (pcorrect + pwrong)
        print '\n', acc, pre, rec, neutral
