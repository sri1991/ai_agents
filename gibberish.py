from words import words_list  # List of English words
import re

# Initialize the English word set (lowercase for comparison)
ENGLISH_WORDS = set(word.lower() for word in words_list)

def is_gibberish(word: str, min_length: int = 3) -> bool:
    """
    Determine if a word is gibberish based on length and character patterns.
    
    Args:
        word (str): The word to check.
        min_length (int): Minimum length for a word to be considered (default: 3).
    
    Returns:
        bool: True if the word is likely gibberish, False otherwise.
    """
    word = word.lower()
    if not word or len(word) < min_length:
        return True
    
    # Check if word is in English dictionary
    if word in ENGLISH_WORDS:
        return False
    
    # Heuristic: Check for repetitive characters (e.g., "aaa", "ababab")
    if len(set(word)) / len(word) < 0.5:
        return True
    
    # Heuristic: Unbalanced vowels/consonants
    vowels = set('aeiou')
    vowel_count = sum(1 for char in word if char in vowels)
    if vowel_count / len(word) < 0.2 or vowel_count / len(word) > 0.8:
        return True
    
    return False

def calculate_text_quality(ocr_text: str) -> tuple[float, float]:
    """
    Calculate the percentage of English text and gibberish text in the OCR output.
    
    Args:
        ocr_text (str): The text extracted from OCR.
    
    Returns:
        tuple[float, float]: (percentage of English text, percentage of gibberish text).
    """
    if not ocr_text or not isinstance(ocr_text, str):
        return 0.0, 0.0

    # Clean the text: remove special characters, normalize spaces
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', ocr_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    if not cleaned_text:
        return 0.0, 0.0

    # Split into words
    tokens = cleaned_text.lower().split()

    if not tokens:
        return 0.0, 0.0

    # Count English, gibberish, and total words
    total_words = len(tokens)
    english_words = sum(1 for word in tokens if word in ENGLISH_WORDS)
    gibberish_words = sum(1 for word in tokens if is_gibberish(word))

    # Calculate percentages
    english_percentage = (english_words / total_words) * 100
    gibberish_percentage = (gibberish_words / total_words) * 100

    # Adjust if percentages exceed 100 due to rounding
    if english_percentage + gibberish_percentage > 100:
        adjustment = (english_percentage + gibberish_percentage - 100) / 2
        english_percentage = max(0, english_percentage - adjustment)
        gibberish_percentage = max(0, gibberish_percentage - adjustment)

    return round(english_percentage, 2), round(gibberish_percentage, 2)

def get_english_and_gibberish_words(ocr_text: str) -> tuple[list, list]:
    """
    Extract English and gibberish words from the OCR output.
    
    Args:
        ocr_text (str): The text extracted from OCR.
    
    Returns:
        tuple[list, list]: (list of English words, list of gibberish words).
    """
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', ocr_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    tokens = cleaned_text.lower().split()
    english_words = [word for word in tokens if word in ENGLISH_WORDS]
    gibberish_words = [word for word in tokens if is_gibberish(word)]
    return english_words, gibberish_words

if __name__ == "__main__":
    # Example usage
    ocr_output = "Hello world! This is a test with some 123 and random t3xt like zxyq! aaa bbbccc ddddd"
    eng_percent, gib_percent = calculate_text_quality(ocr_output)
    eng_words, gib_words = get_english_and_gibberish_words(ocr_output)
    print(f"OCR Text: {ocr_output}")
    print(f"English Words: {eng_words}")
    print(f"Gibberish Words: {gib_words}")
    print(f"Percentage of English Text: {eng_percent}%")
    print(f"Percentage of Gibberish Text: {gib_percent}%")
