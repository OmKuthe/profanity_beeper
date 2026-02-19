from transformers import pipeline

classifier = pipeline("text-classification",
                      model="unitary/toxic-bert")

def detect(words):
    flagged = []

    for w in words:
        result = classifier(w["word"])[0]

        if result["label"] == "toxic":
            flagged.append(w)

    return flagged
