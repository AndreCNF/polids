import difflib
import numpy as np
from sentence_transformers import SentenceTransformer


def remove_formatting_from_text(text: str) -> str:
    """
    Removes all non-alphanumeric characters from a string.
    This makes it easier to check for textual content, while
    disregarding formatting variations, e.g. Markdown syntax.

    Args:
        text (str): The input string.

    Returns:
        str: The input string without any non-alphanumeric
        characters.
    """
    return "".join(filter(str.isalnum, text.lower()))


def compute_semantic_similarity(text1: str, text2: str) -> tuple[float, bool]:
    """
    Computes semantic similarity between two texts using sentence embeddings.
    Works across multiple languages.

    Args:
        text1 (str): First text for comparison
        text2 (str): Second text for comparison

    Returns:
        tuple[float, bool]: A tuple containing:
            - The similarity score (0.0 to 1.0)
            - A boolean indicating if the comparison was successful
    """
    # Lazy-load the model only when needed
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")  # type: ignore

    # Generate embeddings for both texts
    embedding1 = model.encode(text1, convert_to_numpy=True)
    embedding2 = model.encode(text2, convert_to_numpy=True)

    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )
    return similarity, True


def is_text_similar(
    expected: str,
    actual: str,
    difflib_threshold: float = 0.9,
    semantic_threshold: float = 0.8,
) -> bool:
    """
    Checks if two text strings are similar using multiple methods:
    1. Exact containment check
    2. Character-based similarity (difflib)
    3. Semantic similarity (multilingual)

    Args:
        expected (str): The expected text content.
        actual (str): The actual text content to compare against.
        difflib_threshold (float): Minimum character similarity ratio required.
        semantic_threshold (float): Minimum semantic similarity required.

    Returns:
        bool: True if the texts are similar enough by any method, False otherwise.
    """
    # Method 1: Check if expected text is contained within actual text
    expected_clean = remove_formatting_from_text(expected)
    actual_clean = remove_formatting_from_text(actual)

    if expected_clean in actual_clean:
        return True

    # Method 2: Check character-based similarity ratio
    char_similarity = difflib.SequenceMatcher(
        None, expected_clean, actual_clean
    ).ratio()
    if char_similarity >= difflib_threshold:
        return True

    # Method 3: Check semantic similarity (with original formatting)
    semantic_similarity, success = compute_semantic_similarity(expected, actual)
    if success and semantic_similarity >= semantic_threshold:
        return True

    return False
