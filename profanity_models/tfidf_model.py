from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Simple training dataset (replace later with bigger dataset)
texts = [
    "fuck",
    "fucking",
    "have a nice day",
    "damn",
    "Good job",
    "bitch",
    "this is amazing"
]

labels = [1, 1, 0, 1, 0, 1, 0]  # 1 = toxic, 0 = clean

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

def detect(words):
    flagged = []

    for w in words:
        X_test = vectorizer.transform([w["word"]])
        prediction = model.predict(X_test)

        if prediction[0] == 1:
            flagged.append(w)

    return flagged
