from enchant.checker import SpellChecker


def correct_spelling(text):
    checker = SpellChecker('en-US')
    checker.set_text(text)
    corrected = text
    for err in checker:
        if err.word.istitle():
            continue
        index = text.index(err.word)
        if text[index - 1] is '#':
            continue
        sug = checker.suggest(err.word)
        if sug is not None and len(sug) > 0:
            corrected = corrected.replace(err.word, sug[0])
    return corrected
