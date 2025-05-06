import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud  # type: ignore[import]


def get_word_cloud(
    words: list[str],
    max_words: int = 500,
    image_path: str | None = None,
    image_name: str | None = None,
):
    """
    Create a word cloud based on a set of words.

    Args:
        words (list[str]):
            List of words to be included in the word cloud.
        max_words (int):
            Maximum number of words to be included in the word cloud.
        image_path (str):
            Path to the image file where to save the word cloud.
        image_name (str):
            Name of the image where to save the word cloud.
    """

    # Change the value to black
    def black_color_func(
        word, font_size, position, orientation, random_state=None, **kwargs
    ):
        return "hsl(0,100%, 1%)"

    # Set the wordcloud background color to white
    # Set width and height to higher quality, 3000 x 2000
    wordcloud = WordCloud(
        background_color="white",
        width=3000,
        height=2000,
        max_words=max_words,
        stopwords=None,  # We already filtered the stopwords
        regexp=None,  # Just split on whitespace
        min_word_length=3,  # Drop words with less than 3 characters
    ).generate(" ".join(words))
    # Set the word color to black
    wordcloud.recolor(color_func=black_color_func)
    # Set the figsize
    plt.figure(figsize=(15, 10))
    # Plot the wordcloud
    plt.imshow(wordcloud, interpolation="bilinear")
    # Remove plot axes
    plt.axis("off")
    if image_path is not None and image_name is not None:
        # Save the image
        plt.savefig(os.path.join(image_path, image_name), bbox_inches="tight")
