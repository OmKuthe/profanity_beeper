import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

class TFIDFProfanityDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, lowercase=True)
        self.classifier = LogisticRegression()
        self.lemmatizer = WordNetLemmatizer()
        self.is_trained = False
        
        # Try to load pre-trained model
        model_path = os.path.join(os.path.dirname(__file__), 'tfidf_model.pkl')
        if os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Train on some default data
            self._train_default()
    
    def _train_default(self):
        """Train on default dataset"""
        # Sample training data
        train_texts = [
            "this is a damn good movie",
            "what the hell is going on",
            "i love this movie",
            "this is great",
            "that's shit",
            "you are awesome",
            "fuck you",
            "have a nice day",
            "bloody hell",
            "wonderful performance"
        ]
        train_labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
        
        # Train
        X = self.vectorizer.fit_transform(train_texts)
        self.classifier.fit(X, train_labels)
        self.is_trained = True
    
    def lemmatize_text(self, text):
        """Apply lemmatization to text"""
        words = text.split()
        lemmatized = [self.lemmatizer.lemmatize(w.lower()) for w in words]
        return ' '.join(lemmatized)
    
    def detect(self, words, use_lemmas=True):
        """
        Detect profanity using TF-IDF model
        """
        flagged = []
        
        # Convert words to text
        if isinstance(words, list):
            if all(isinstance(w, dict) for w in words):
                # Extract words from dictionaries
                text_words = [w.get('word', '') for w in words]
            else:
                text_words = words
            
            text = ' '.join(text_words)
        else:
            text = words
        
        # Apply lemmatization if requested
        if use_lemmas:
            text = self.lemmatize_text(text)
        
        # Transform and predict
        if self.is_trained and text.strip():
            X = self.vectorizer.transform([text])
            proba = self.classifier.predict_proba(X)[0]
            
            # If probability of profanity > 0.5, flag all words? 
            # For simplicity, we'll flag based on keyword matching in this demo
            if proba[1] > 0.5:
                # Fall back to keyword detection for individual words
                from .keyword_filter import detect as keyword_detect
                flagged = keyword_detect(words, use_lemmas=use_lemmas)
        
        return flagged
    
    def save_model(self, path):
        """Save trained model"""
        joblib.dump({
            'vectorizer': self.vectorizer,
            'classifier': self.classifier
        }, path)
    
    def load_model(self, path):
        """Load trained model"""
        data = joblib.load(path)
        self.vectorizer = data['vectorizer']
        self.classifier = data['classifier']
        self.is_trained = True

# Create global instance
_detector = TFIDFProfanityDetector()

def detect(words, use_lemmas=True):
    """Main detection function"""
    return _detector.detect(words, use_lemmas)