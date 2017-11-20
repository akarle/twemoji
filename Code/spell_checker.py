from enchant.checker import SpellChecker


def correct_spelling(data):
    checker = SpellChecker('en-US')
    corrections = []
    count = 0
    for text in data:
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
        corrections.append(corrected)
        count += 1
        if count > 1 and count % 100 == 0:
            print "Spelling checking " + str(float(count) /
                                             len(data) * 100) + "% done..."
    return corrections
