import pytest

from polids.utils.text_similarity import (  # type: ignore[import]
    remove_formatting_from_text,
    compute_semantic_similarity,
    compute_text_similarity_scores,
    is_text_similar,
)


def test_remove_formatting_from_text() -> None:
    """
    Test the removal of non-alphanumeric characters from text.
    """
    assert remove_formatting_from_text("Hello, World!") == "helloworld", (
        "Expected 'helloworld' after removing formatting, but got a different result."
    )
    assert remove_formatting_from_text("123-456-7890") == "1234567890", (
        "Expected '1234567890' after removing formatting, but got a different result."
    )
    assert remove_formatting_from_text("Markdown **syntax**") == "markdownsyntax", (
        "Expected 'markdownsyntax' after removing formatting, but got a different result."
    )
    assert remove_formatting_from_text("") == "", (
        "Expected an empty string after removing formatting, but got a different result."
    )


@pytest.mark.parametrize(
    "text1, text2, comparison_text1, comparison_text2",
    [
        ("Hello world", "Hello world", "Hello", "Hi"),  # Same text vs synonyms
        ("Hello", "Hi", "Happy", "Sad"),  # Synonym vs antonym
        (
            "Happy",
            "Sad",
            "The quick brown fox jumps over the lazy dog.",
            "1234567890 is a sequence of digits.",
        ),  # Antonym vs unrelated
    ],
)
def test_compute_semantic_similarity_relative(
    text1: str, text2: str, comparison_text1: str, comparison_text2: str
) -> None:
    """
    Test relative semantic similarity between pairs of texts.
    """
    similarity_1 = compute_semantic_similarity(text1, text2)
    similarity_2 = compute_semantic_similarity(comparison_text1, comparison_text2)
    assert similarity_1 > similarity_2, (
        f"Expected similarity of '{text1}' and '{text2}' ({similarity_1}) to be greater "
        f"than similarity of '{comparison_text1}' and '{comparison_text2}' ({similarity_2})."
    )


@pytest.mark.parametrize(
    "expected, actual, comparison_expected, comparison_actual",
    [
        ("Hello world", "Hello world", "Hello", "Hi"),  # Same text vs synonyms
        ("Hello", "Hi", "Happy", "Sad"),  # Synonym vs antonym
        (
            "Happy",
            "Sad",
            "The quick brown fox jumps over the lazy dog.",
            "1234567890 is a sequence of digits.",
        ),  # Antonym vs unrelated
    ],
)
def test_compute_text_similarity_scores_relative(
    expected: str, actual: str, comparison_expected: str, comparison_actual: str
) -> None:
    """
    Test relative character-based and semantic similarity scores between pairs of texts.
    """
    char_sim_1, sem_sim_1 = compute_text_similarity_scores(expected, actual)
    char_sim_2, sem_sim_2 = compute_text_similarity_scores(
        comparison_expected, comparison_actual
    )
    assert sem_sim_1 > sem_sim_2, (
        f"Expected semantic similarity of '{expected}' and '{actual}' ({sem_sim_1}) to be greater "
        f"than semantic similarity of '{comparison_expected}' and '{comparison_actual}' ({sem_sim_2})."
    )
    assert char_sim_1 > char_sim_2, (
        f"Expected character similarity of '{expected}' and '{actual}' ({char_sim_1}) to be greater "
        f"than character similarity of '{comparison_expected}' and '{comparison_actual}' ({char_sim_2})."
    )


@pytest.mark.parametrize(
    "expected, actual, difflib_threshold, semantic_threshold, expected_result",
    [
        ("Hello world", "Hello world", 1, 1, True),
        ("Hello", "Hi", 0.9, 0.8, True),
        ("Happy", "Sad", 0.5, 0.5, True),  # Low thresholds
        ("Happy", "Sad", 0.9, 0.99, False),  # High thresholds
        (
            "The quick brown fox jumps over the lazy dog.",
            "1234567890 is a sequence of digits.",
            0.9,
            0.8,
            False,  # Unrelated texts
        ),
    ],
)
def test_is_text_similar(
    expected: str,
    actual: str,
    difflib_threshold: float,
    semantic_threshold: float,
    expected_result: bool,
) -> None:
    """
    Test if two texts are considered similar based on thresholds.
    """
    result = is_text_similar(expected, actual, difflib_threshold, semantic_threshold)
    assert result == expected_result, (
        f"Expected similarity result of {expected_result} for texts '{expected}' and "
        f"'{actual}' with thresholds (difflib: {difflib_threshold}, semantic: {semantic_threshold}), "
        f"but got {result}."
    )
