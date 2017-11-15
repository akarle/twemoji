# need to update parse_loadout, run_me and feat_extract to use list of CV args instead of analyzer
# pos tagging option in run_me
# cv = CountVectorizer(token_pattern=r'(?u)\bpos_._\w+\b')
from CMUTweetTagger import runtagger_parse


def pos_tag(data):
    """
    data: list of strings
    returns: list of POS tagged strings
    """
    return [" ".join(['POS_' + il[1] + '_' +
            il[0] for il in ol]) for ol in runtagger_parse(data)]
