import pytest
from pathlib import Path

from polids.utils.data_viz import get_word_cloud  # type: ignore[import]


@pytest.fixture
def sample_words() -> list[str]:
    """Fixture providing a sample list of words."""
    return ["python", "data", "visualization", "word", "cloud", "test", "pytest"]


@pytest.fixture
def temp_image_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory for saving images."""
    return tmp_path


def test_get_word_cloud_without_saving(sample_words: list[str]) -> None:
    """Test generating a word cloud without saving it to a file."""
    try:
        get_word_cloud(words=sample_words)
    except Exception as e:
        pytest.fail(f"Word cloud generation failed with error: {e}")


def test_get_word_cloud_with_saving(
    sample_words: list[str], temp_image_dir: Path
) -> None:
    """Test generating a word cloud and saving it to a file."""
    image_name = "test_wordcloud.png"
    image_path = temp_image_dir / image_name

    get_word_cloud(
        words=sample_words,
        image_path=str(temp_image_dir),
        image_name=image_name,
    )

    assert image_path.exists(), f"Expected image file {image_name} was not created."
    assert image_path.stat().st_size > 0, "Generated image file is empty."
