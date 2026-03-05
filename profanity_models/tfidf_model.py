from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# =====================================
# SIMPLE TRAINING DATASET
# =====================================
# Profane words (label = 1)
profane_words = [
    "fuck", "fucking", "fucked", "fucker", "fuckers", "fuckin",
    "shit", "shitting", "shitted", "bullshit", "shitty",
    "damn", "damned", "goddamn", "dammit",
    "bitch", "bitches", "bitching",
    "ass", "asshole", "asses",
    "cunt", "pussy", "dick", "cock",
    "bastard", "bastards",
    "whore", "slut", "hooker",
    "piss", "pissing", "pissed",
    "motherfucker", "motherfucking",
]

# Clean words (label = 0) - just a few common ones
clean_words = [
    "hello", "hi", "hey",
    "please", "thank", "thanks",
    "the", "a", "an",
    "and", "or", "but",
    "is", "are", "was",
    "good", "bad", "nice",
    "you", "your", "i", "me",
    "can", "will", "would",
    "up", "down", "here", "there","like","patrick","shut"
]

# Combine and create labels
texts = profane_words + clean_words
labels = [1] * len(profane_words) + [0] * len(clean_words)

print(f"Training samples: {len(texts)} total")
print(f"  - Profane words: {len(profane_words)}")
print(f"  - Clean words: {len(clean_words)}")

# =====================================
# VECTORIZER & MODEL
# =====================================
vectorizer = TfidfVectorizer(
    lowercase=True,
    analyzer='char_wb',
    ngram_range=(2, 4),
    min_df=1,
)

X = vectorizer.fit_transform(texts)

model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42,
    class_weight='balanced'  # Handle any imbalance
)
model.fit(X, labels)

# =====================================
# DETECTION FUNCTION
# =====================================
def detect(words):
    """
    Args:
        words: List of dicts [{"word": str, "start": float, "end": float}, ...]
    
    Returns:
        List of dicts for profane words (same format as input)
    """
    flagged = []
    
    if not words:
        return flagged
    
    # Transform all words
    word_texts = [w["word"].lower() for w in words]
    X_test = vectorizer.transform(word_texts)
    
    # Get predictions
    predictions = model.predict(X_test)
    
    # Flag only the profane ones
    for i, w in enumerate(words):
        if predictions[i] == 1:
            flagged.append(w)
    
    return flagged

# Quick test
if __name__ == "__main__":
    test_words = [
        {"word": "fuck", "start": 0, "end": 1},
        {"word": "hello", "start": 1, "end": 2},
        {"word": "damn", "start": 2, "end": 3},
        {"word": "please", "start": 3, "end": 4},
        {"word": "bitch", "start": 4, "end": 5},
        {"word": "the", "start": 5, "end": 6},
        {"word": "shit", "start": 6, "end": 7},
    ]
    
    results = detect(test_words)
    print("Flagged words:", [w["word"] for w in results])
    
    # Should print: ['fuck', 'damn', 'bitch', 'shit']