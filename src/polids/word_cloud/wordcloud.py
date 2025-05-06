from polids.utils.linguistic_analysis import get_clean_words_from_text
from polids.utils.data_viz import get_word_cloud


class WordCloudGenerator:
    """
    Class to generate a word cloud from a given text.
    """

    def __init__(self, text: str):
        """
        Initialize the WordCloudGenerator with the given text.

        Args:
            text (str): The text to analyze.
        """
        self.text = text
        self.words = get_clean_words_from_text(text)

    def generate_word_cloud(
        self, image_path: str | None, image_name: str | None
    ) -> None:
        """
        Generate and save a word cloud image from the cleaned words.

        Args:
            image_path (str): The path where to save the word cloud image.
            image_name (str): The name of the word cloud image file.
        """
        get_word_cloud(self.words, image_path=image_path, image_name=image_name)
