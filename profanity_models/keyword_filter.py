def load_bad_words():
    with open("badwords.txt") as f:
        return set(w.strip().lower() for w in f.readlines())

BAD_WORDS = load_bad_words()

def detect(words):
    flagged = []

    for w in words:
        if w["word"] in BAD_WORDS:
            flagged.append(w)

    return flagged
