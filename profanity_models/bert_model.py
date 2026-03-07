import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
import os

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

class BERTProfanityDetector:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        # In a real implementation, you'd load a BERT model
        # For demo, we'll simulate BERT with advanced detection
        print("BERT Model initialized (simulated)")
        
        # Expanded profanity database with contextual understanding
        self.profanity_db = {
            'strong': {'fuck', 'shit', 'cunt', 'motherfucker'},
            'medium': {'damn', 'hell', 'bitch', 'asshole', 'bastard'},
            'mild': {'crap', 'piss', 'darn', 'heck', 'bloody'}
        }
        
        # Lemmatized versions
        self.lemmatized_db = {}
        for level, words in self.profanity_db.items():
            self.lemmatized_db[level] = set()
            for word in words:
                if word.isalpha():
                    self.lemmatized_db[level].add(self.lemmatizer.lemmatize(word))
    
    def detect(self, words, use_lemmas=True):
        """
        Detect profanity using BERT (simulated)
        Uses contextual understanding and intensity levels
        """
        flagged = []
        
        # Process each word with context window
        for i, word_dict in enumerate(words):
            if isinstance(word_dict, dict):
                original = word_dict.get('word', '')
                word = original.lower()
                start_time = word_dict.get('start', 0)
                end_time = word_dict.get('end', 0)
            else:
                original = word_dict
                word = original.lower()
                start_time = 0
                end_time = 0
            
            # Get context (words before and after)
            context_words = []
            for j in range(max(0, i-2), min(len(words), i+3)):
                if j != i:
                    if isinstance(words[j], dict):
                        context_words.append(words[j].get('word', '').lower())
                    else:
                        context_words.append(words[j].lower())
            
            # Determine if word is profane
            is_profane = False
            intensity = None
            
            if use_lemmas and word.isalpha():
                lemma = self.lemmatizer.lemmatize(word)
                
                # Check each intensity level
                for level, lemmas in self.lemmatized_db.items():
                    if lemma in lemmas:
                        is_profane = True
                        intensity = level
                        break
            else:
                # Check original word
                for level, words_set in self.profanity_db.items():
                    if word in words_set:
                        is_profane = True
                        intensity = level
                        break
            
            # Contextual analysis (simulated BERT attention)
            if is_profane:
                # Calculate confidence based on context
                confidence = 0.95  # Base confidence for BERT
                
                # Adjust based on intensity
                if intensity == 'strong':
                    confidence *= 1.0
                elif intensity == 'medium':
                    confidence *= 0.9
                else:  # mild
                    confidence *= 0.8
                
                # Check if context suggests non-profane usage
                non_profane_context = {
                    'heck': ['what the', 'oh'],
                    'dam': ['beaver', 'wall'],
                    'ass': ['class', 'mass']
                }
                
                context_str = ' '.join(context_words)
                for word_check, contexts in non_profane_context.items():
                    if word == word_check and any(ctx in context_str for ctx in contexts):
                        confidence *= 0.3  # Reduce confidence
                
                # Flag if confidence is high enough
                if confidence > 0.6:
                    # Create enhanced word dict with BERT info
                    enhanced_word = word_dict.copy() if isinstance(word_dict, dict) else {
                        'word': word_dict,
                        'start': start_time,
                        'end': end_time
                    }
                    enhanced_word['bert_confidence'] = confidence
                    enhanced_word['intensity'] = intensity
                    enhanced_word['context'] = context_words
                    
                    flagged.append(enhanced_word)
        
        return flagged

# Create global instance
_detector = BERTProfanityDetector()

def detect(words, use_lemmas=True):
    """Main detection function"""
    return _detector.detect(words, use_lemmas)