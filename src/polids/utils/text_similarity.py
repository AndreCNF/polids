import difflib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from loguru import logger


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


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Computes semantic similarity between two texts using sentence embeddings.
    Works across multiple languages.

    Args:
        text1 (str): First text for comparison
        text2 (str): Second text for comparison

    Returns:
        float: Cosine similarity score between the two texts, between 0 and 1.
    """
    # Lazy-load the model: prefer MPS → TPU → CUDA → CPU
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        try:
            import importlib

            xm = importlib.import_module("torch_xla.core.xla_model")
            device = str(xm.xla_device())
        except ImportError:
            if torch.cuda.is_available():
                device = "cuda"
    model = SentenceTransformer("all-MiniLM-L12-v2", device=device)  # type: ignore

    # Generate embeddings for both texts, with fallback to CPU on MPS OOM
    try:
        embedding1 = model.encode(text1, convert_to_numpy=True)
        embedding2 = model.encode(text2, convert_to_numpy=True)
    except RuntimeError as e:
        if torch.backends.mps.is_available() and "memory" in str(e).lower():
            logger.warning(
                "MPS out of memory encountered; falling back to CPU for encoding."
            )
            embedding1 = model.encode(text1, convert_to_numpy=True, device="cpu")
            embedding2 = model.encode(text2, convert_to_numpy=True, device="cpu")
        else:
            raise

    # Clear device cache to free up memory
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )

    # Normalize the similarity score to be between 0 and 1
    similarity = (similarity + 1) / 2

    return similarity


def compute_text_similarity_scores(
    expected: str,
    actual: str,
) -> tuple[float, float]:
    """
    Computes various similarity scores between two text strings.

    Args:
        expected (str): The expected text content.
        actual (str): The actual text content to compare against.

    Returns:
        tuple[float, float]: A tuple containing:
            - Character-based similarity score
            - Semantic similarity score
    """
    # Clean the texts for character-based similarity
    expected_clean = remove_formatting_from_text(expected)
    actual_clean = remove_formatting_from_text(actual)

    if expected_clean in actual_clean:
        # If the expected text is contained in the actual text,
        # return maximum similarity scores
        return 1.0, 1.0

    # Calculate character-based similarity
    char_similarity = difflib.SequenceMatcher(
        None, expected_clean, actual_clean
    ).ratio()

    # Calculate semantic similarity
    semantic_similarity = compute_semantic_similarity(expected, actual)

    return char_similarity, semantic_similarity


def is_text_similar(
    expected: str,
    actual: str,
    difflib_threshold: float = 0.8,
    semantic_threshold: float = 0.9,
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
    char_similarity, semantic_similarity = compute_text_similarity_scores(
        expected, actual
    )
    if (
        char_similarity >= difflib_threshold
        or semantic_similarity >= semantic_threshold
    ):
        return True
    else:
        return False
