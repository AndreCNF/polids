from typing import List, Tuple
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy

SPACY_NLP = spacy.load("pt_core_news_lg")


def get_word_cloud(words: List[str], max_words=500, image_path=None, image_name=None):
    """
    Create a word cloud based on a set of words.

    Args:
        words (List[str]):
            List of words to be included in the word cloud.
        max_words (int):
            Maximum number of words to be included in the word cloud.
        image_path (str):
            Path to the image file where to save the word cloud.
        image_name (str):
            Name of the image where to save the word cloud.
    """
    # change the value to black
    def black_color_func(
        word, font_size, position, orientation, random_state=None, **kwargs
    ):
        return "hsl(0,100%, 1%)"

    # set the wordcloud background color to white
    # set width and height to higher quality, 3000 x 2000
    wordcloud = WordCloud(
        font_path="/Library/Fonts/Arial Unicode.ttf",
        background_color="white",
        width=3000,
        height=2000,
        max_words=max_words,
    ).generate(" ".join(words))
    # set the word color to black
    wordcloud.recolor(color_func=black_color_func)
    # set the figsize
    plt.figure(figsize=[15, 10])
    # plot the wordcloud
    plt.imshow(wordcloud, interpolation="bilinear")
    # remove plot axes
    plt.axis("off")
    if image_path is not None and image_name is not None:
        # save the image
        plt.savefig(os.path.join(image_path, image_name), bbox_inches="tight")


def _get_phrases(doc: spacy.tokens.doc.Doc) -> List[str]:
    """
    Get phrases from a text. Also remove new line symbols.

    Args:
        doc (spacy.tokens.doc.Doc):
            Spacy document.

    Returns:
        List[str]:
            List of phrases.
    """
    phrases = [sent.text.replace("\n", "") for sent in doc.sents]
    return phrases


def _get_words(doc: spacy.tokens.doc.Doc) -> List[str]:
    """
    Get every word in the text that isn't a stopword or punctuation,
    and that is either a noun, adjective, verb or interjection
    (based on the [universal POS tags](https://universaldependencies.org/u/pos/))

    Args:
        doc (spacy.tokens.doc.Doc):
            Spacy document.

    Returns:
        List[str]:
            List of words.
    """
    words = [
        word.text.replace("\n", "")  # remove new line symbols
        for word in doc
        if not word.is_stop  # remove stopwords
        and not word.is_punct  # remove punctuation
        and (
            word.pos_ == "NOUN"  # noun
            or word.pos_ == "ADJ"  # adjective
            or word.pos_ == "VERB"  # verb
            or word.pos_ == "INTJ"  # interjection
            or word.pos_ == "X"  # other
        )
    ]
    # remove blank words
    words = [word for word in words if word != ""]
    return words


def get_phrases_and_words(text: str) -> Tuple[List[str], List[str]]:
    """
    Get phrases and words from a text.

    Args:
        text (str):
            Text to be processed.

    Returns:
        List[str]:
            List of phrases.
        List[str]:
            List of words.
    """
    doc = SPACY_NLP(text)
    phrases = _get_phrases(doc)
    words = _get_words(doc)
    return phrases, words
