from pydantic import BaseModel, Field  # type: ignore[import]
from pydantic_extra_types.language_code import LanguageAlpha2, LanguageName  # type: ignore[import]
from lingua import LanguageDetectorBuilder
import pycountry
import string
import nltk  # type: ignore[import]
from nltk.corpus import stopwords  # type: ignore[import]
from nltk.tokenize import word_tokenize  # type: ignore[import]
from nltk.stem import SnowballStemmer  # type: ignore[import]
import simplemma


class LanguageOfText(BaseModel):
    """
    Language of a given text.
    """

    name: LanguageName = Field(
        description="Full worded name of the language, e.g. 'English'."
    )
    code: LanguageAlpha2 = Field(
        description="Two-letter code of the language, e.g. 'en' for English."
    )


# Include only languages that are not yet extinct (= currently excludes Latin)
LANGUAGE_DETECTOR = LanguageDetectorBuilder.from_all_spoken_languages().build()

# Prepare nltk for stopword removal, tokenization and stemming
nltk.download("stopwords")
nltk.download("punkt_tab")


def detect_language(text: str) -> LanguageOfText:
    """
    Detect the language of a given text.

    Args:
        text (str): The text to analyze.

    Returns:
        LanguageOfText: The detected language and its Alpha-2 code.

    Raises:
        ValueError: If the language detection fails.
    """
    # Detect the language using Lingua
    language = LANGUAGE_DETECTOR.detect_language_of(text)

    if not language:
        raise ValueError("Language detection failed.")

    # Get the language name and code
    language_name = language.name.lower()
    language_code = pycountry.languages.get(name=language_name.capitalize()).alpha_2

    return LanguageOfText(
        name=language_name,  # type: ignore
        code=language_code,
    )


def get_clean_words_from_text(text: str) -> list[str]:
    """
    Get clean words from a given text, i.e. remove stopwords, punctuation, and lemmatize.

    Args:
        text (str): The text to analyze.

    Returns:
        list[str]: A list of cleaned words.
    """
    # Detect the language of the text
    language = detect_language(text)
    # Get the stopwords for the detected language
    stop_words = set(stopwords.words(language.name))
    # Split the text into words
    word_tokens = word_tokenize(text)
    # Lowercase and trim words
    word_tokens_clean = [w.lower().strip() for w in word_tokens]
    # Filter and clean words
    filtered_words = [
        # Convert to lower case and lemmatize the words
        simplemma.lemmatize(w.lower(), lang=language.code)
        for w in word_tokens_clean
        # Remove stopwords
        if w.lower() not in stop_words
        # Remove punctuation and markdown symbols
        and w not in string.punctuation
        and not w.startswith("#")
        # Remove words with numbers
        and not any(char.isdigit() for char in w)
    ]
    # Stem words if they have a special character
    stemmer = SnowballStemmer(language.name)
    filtered_words = [
        stemmer.stem(w) if any(char in string.punctuation for char in w) else w
        for w in filtered_words
    ]
    # Remove words with less than 3 characters
    filtered_words = [w for w in filtered_words if len(w) > 2]
    return filtered_words
