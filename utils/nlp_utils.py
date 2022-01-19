from typing import Dict, List
import os
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import spacy
from string import punctuation
import pandas as pd

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


def _add_sentence_to_list(sentence: str, sentences_list: List[str]):
    """
    Add a sentence to the list of sentences.

    Args:
        sentence (str):
            Sentence to be added.
        sentences (List[str]):
            List of sentences.
    """
    while sentence.startswith(" "):
        # remove leading space
        sentence = sentence[1:]
    if all(c in punctuation for c in sentence) or len(sentence) == 1:
        # skip sentences with only punctuation
        return
    sentences_list.append(sentence)


def get_sentences(text: str) -> List[str]:
    """
    Get sentences from a text.

    Args:
        text (str):
            Text to be processed.

    Returns:
        List[str]:
            List of sentences.
    """
    # get the paragraphs
    paragraphs = text.split("\n")
    paragraphs = [p for p in paragraphs if p != ""]
    # get the sentences from the paragraphs
    sentences = list()
    for paragraph in paragraphs:
        if paragraph.startswith("#"):
            _add_sentence_to_list(paragraph, sentences)
            continue
        prev_sentence_idx = 0
        for idx in range(len(paragraph)):
            if idx + 1 < len(paragraph):
                if (paragraph[idx] == "." and not paragraph[idx + 1].isdigit()) or (
                    paragraph[idx] in "!?"
                ):
                    sentence = paragraph[prev_sentence_idx : idx + 1]
                    _add_sentence_to_list(sentence, sentences)
                    prev_sentence_idx = idx + 1
            else:
                sentence = paragraph[prev_sentence_idx:]
                _add_sentence_to_list(sentence, sentences)
    return sentences


def get_words(text: str) -> List[str]:
    """
    Get every word in the text that isn't a stopword or punctuation,
    and that is either a noun, adjective, verb or interjection
    (based on the [universal POS tags](https://universaldependencies.org/u/pos/))

    Args:
        text (str):
            Text to be processed.

    Returns:
        List[str]:
            List of words.
    """
    doc = SPACY_NLP(text)
    words = [
        word.text.replace("\n", "").replace("*", "")  # remove new line and bold symbols
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


def get_topical_sentences(
    sentences: List[str], topics: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Get lists of sentences per topic.

    Args:
        sentences (List[str]):
            List of sentences to analyse.
        topics (Dict[str, List[str]]):
            Dictionary of words per topic.

    Returns:
        Dict[str, List[str]]:
            Dictionary of sentences per topic.
    """
    topical_sentences = dict()
    for topic in topics:
        topical_sentences[topic] = list()
    for sentence in sentences:
        for topic in topics:
            if any(topical_word in sentence for topical_word in topics[topic]):
                topical_sentences[topic].append(sentence)
    return topical_sentences


def plot_topical_presence(
    sentences: List[str], topics: Dict[str, List[str]], title: str = None
):
    """
    Plot the number of sentences per topic.

    Args:
        sentences (List[str]):
            List of sentences to analyse.
        topics (Dict[str, List[str]]):
            Dictionary of words per topic.
    """
    topical_sentences = get_topical_sentences(sentences, topics)
    topic_sentence_count = dict()
    for topic in topical_sentences.keys():
        topic_sentence_count[topic] = len(topical_sentences[topic])
    topic_sentence_count = pd.DataFrame(
        topic_sentence_count, index=["sentence_count"]
    ).T
    topic_sentence_count["sentence_percentage"] = (
        topic_sentence_count["sentence_count"] / len(sentences) * 100
    )
    topic_sentence_count.index.name = "topic"
    topic_sentence_count.sort_index(inplace=True)
    fig = px.bar(topic_sentence_count, x="sentence_percentage", orientation="h")
    fig.update_layout(
        title=title,
        xaxis_title="Percentagem de frases topicais no texto",
        yaxis_title="Tópico",
        yaxis=dict(categoryorder="category descending"),
        # marker_color="rgb(0, 0, 0)",
    )
