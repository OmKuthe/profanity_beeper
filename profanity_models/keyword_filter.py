import nltk
from nltk.stem import WordNetLemmatizer
import ssl

# Fix SSL for NLTK download
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Initialize lemmatizer
_lemmatizer = WordNetLemmatizer()

# Original bad words list
BAD_WORDS = {
    'damn', 'hell', 'shit', 'fuck', 'ass', 'bitch', 'crap',
    'darn', 'heck', 'bloody', 'piss', 'cock', 'dick', 'pussy',
    'bastard', 'slut', 'whore', 'douche', 'fag', 'dyke',
    # Add more as needed
}

# Create lemmatized version for better matching
LEMMATIZED_BAD_WORDS = set()
for word in BAD_WORDS:
    if word.isalpha():
        LEMMATIZED_BAD_WORDS.add(_lemmatizer.lemmatize(word.lower()))

def detect(words, use_lemmas=True):
    """
    Detect profanity in words
    
    Args:
        words: List of word dictionaries with 'word' key
        use_lemmas: If True, use lemmatized comparison for better matching
    
    Returns:
        List of flagged word dictionaries
    """
    flagged = []
    
    for word_dict in words:
        # Handle both string words and dictionary formats
        if isinstance(word_dict, dict):
            original_word = word_dict.get('word', '')
            # If lemmatized version is provided, use it
            lemma = word_dict.get('lemma', original_word.lower())
        else:
            original_word = word_dict
            lemma = original_word.lower()
        
        # Check if word is profane
        is_profane = False
        
        if use_lemmas:
            # Lemmatize for comparison
            if lemma.isalpha():
                # If lemma already provided, use it; otherwise lemmatize
                if lemma == original_word.lower():  # No lemma provided
                    lemma = _lemmatizer.lemmatize(original_word.lower())
                
                if lemma in LEMMATIZED_BAD_WORDS:
                    is_profane = True
        else:
            # Simple lowercase check
            if original_word.lower() in BAD_WORDS:
                is_profane = True
        
        if is_profane:
            flagged.append(word_dict)
    
    return flagged

# For backward compatibility
def detect_simple(words):
    """Simple detection without lemmas"""
    return detect(words, use_lemmas=False)