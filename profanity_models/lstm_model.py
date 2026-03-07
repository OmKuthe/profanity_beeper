import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
import os
import json

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

class LSTMProfanityDetector:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        # In a real implementation, you'd load a trained LSTM model
        # For demo, we'll simulate LSTM with enhanced keyword detection
        print("LSTM Model initialized (simulated)")
    
    def preprocess(self, words, use_lemmas=True):
        """Preprocess words for LSTM"""
        processed = []
        for w in words:
            if isinstance(w, dict):
                word = w.get('word', '')
                if use_lemmas and word.isalpha():
                    lemma = w.get('lemma', self.lemmatizer.lemmatize(word.lower()))
                    processed.append(lemma)
                else:
                    processed.append(word.lower())
            else:
                processed.append(w.lower())
        return processed
    
    def detect(self, words, use_lemmas=True):
        """
        Detect profanity using LSTM (simulated)
        For demo, we'll use enhanced detection with context
        """
        flagged = []
        
        # Preprocess words
        processed_words = self.preprocess(words, use_lemmas)
        
        # Context-aware profanity words (expanded)
        profane_base = {
            'damn', 'hell', 'shit', 'fuck', 'ass', 'bitch', 'crap',
            'darn', 'heck', 'bloody', 'piss', 'cock', 'dick', 'pussy',
            'bastard', 'slut', 'whore', 'douche'
        }
        
        # Lemmatize profane words for matching
        profane_lemmas = set()
        for word in profane_base:
            if word.isalpha():
                profane_lemmas.add(self.lemmatizer.lemmatize(word))
        
        # Check each word with context
        for i, word_dict in enumerate(words):
            if isinstance(word_dict, dict):
                original = word_dict.get('word', '')
                word = original.lower()
            else:
                original = word_dict
                word = original.lower()
            
            # Check if word is profane
            is_profane = False
            
            if use_lemmas and word.isalpha():
                lemma = self.lemmatizer.lemmatize(word)
                if lemma in profane_lemmas:
                    is_profane = True
            elif word in profane_base:
                is_profane = True
            
            # Context check (simulated LSTM behavior)
            if is_profane:
                # Check surrounding words for context (simplified)
                context_before = ""
                context_after = ""
                
                if i > 0:
                    prev = words[i-1]
                    if isinstance(prev, dict):
                        context_before = prev.get('word', '').lower()
                    else:
                        context_before = prev.lower()
                
                if i < len(words) - 1:
                    nxt = words[i+1]
                    if isinstance(nxt, dict):
                        context_after = nxt.get('word', '').lower()
                    else:
                        context_after = nxt.lower()
                
                # Simulate confidence based on context
                confidence = 0.9  # Base confidence
                
                # Lower confidence if it might be a false positive
                false_positive_indicators = ['heck', 'dam', 'assess', 'classic']
                if word in false_positive_indicators:
                    confidence *= 0.5
                
                # Add to flagged if confidence > threshold
                if confidence > 0.6:
                    flagged.append(word_dict)
        
        return flagged

# Create global instance
_detector = LSTMProfanityDetector()

def detect(words, use_lemmas=True):
    """Main detection function"""
    return _detector.detect(words, use_lemmas)