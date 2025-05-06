import pytest

from polids.utils.linguistic_analysis import (  # type: ignore[import]
    detect_language,
    get_clean_words_from_text,
    LanguageOfText,
)


@pytest.mark.parametrize(
    "text, expected_language_name, expected_language_code",
    [
        ("This is a test sentence.", "English", "en"),
        ("Dies ist ein Testsatz.", "German", "de"),
        ("Esta es una oración de prueba.", "Spanish", "es"),
        ("Esta é uma frase de teste.", "Portuguese", "pt"),
    ],
)
def test_detect_language(
    text: str, expected_language_name: str, expected_language_code: str
):
    """
    Test the detect_language function with multiple languages.
    """
    detected_language = detect_language(text)
    assert isinstance(detected_language, LanguageOfText), (
        f"Expected LanguageOfText, got {type(detected_language)}."
    )
    assert detected_language.name == expected_language_name, (
        f"Expected language name '{expected_language_name}', got '{detected_language.name}'."
    )
    assert detected_language.code == expected_language_code, (
        f"Expected language code '{expected_language_code}', got '{detected_language.code}'."
    )


@pytest.mark.parametrize(
    "text, expected_clean_words",
    [
        ("This is a test sentence.", ["test", "sentence"]),
        ("Dies ist ein Testsatz.", ["testsatz"]),
        ("Esta es una oración de prueba.", ["oración", "prueba"]),
        ("Esta é uma frase de teste.", ["frase", "teste"]),
    ],
)
def test_get_clean_words_from_text(text: str, expected_clean_words: list[str]):
    """
    Test the get_clean_words_from_text function with multiple languages.
    """
    clean_words = get_clean_words_from_text(text)
    assert clean_words == expected_clean_words, (
        f"Expected clean words '{expected_clean_words}', got '{clean_words}'."
    )


def test_detect_language_invalid_input():
    """
    Test the detect_language function with invalid input.
    """
    with pytest.raises(ValueError, match="Language detection failed."):
        detect_language("1234567890")


def test_get_clean_words_from_text_empty_input():
    """
    Test the get_clean_words_from_text function with an empty string.
    """
    assert get_clean_words_from_text("") == [], (
        "Expected an empty list for empty input."
    )
