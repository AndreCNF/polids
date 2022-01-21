from typing import Dict, List
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure
import os
import pandas as pd

from utils.nlp_utils import get_topical_sentences


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


def plot_topical_presence(
    sentences: List[str],
    topics: Dict[str, List[str]],
    title: str = None,
    color: str = "blue",
) -> Figure:
    """
    Plot the number of sentences per topic.

    Args:
        sentences (List[str]):
            List of sentences to analyse.
        topics (Dict[str, List[str]]):
            Dictionary of words per topic.
        title (str):
            Title of the plot.
        color (str):
            Color of the bars in the plot.

    Returns:
        Figure:
            Plotly figure with the number of sentences per topic.
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
    )
    fig.update_traces(marker_color=color)
    return fig


def plot_approaches(sentences: List[str], approaches: Dict[str, List[str]]) -> Figure:
    """
    Plot the approaches taken to language and policy.

    Args:
        sentences (List[str]):
            List of sentences to analyse.
        approaches (Dict[str, List[str]]):
            Dictionary of words per approach.

    Returns:
        Figure:
            Plotly figure with the number of sentences per approach.
    """
    approach_sentences = get_topical_sentences(sentences, approaches)
    approach_sentence_count = dict()
    total_num_sentences_in_approaches = sum(
        [len(approach_sentences[approach]) for approach in approach_sentences.keys()]
    )
    for approach in approaches:
        approach_sentence_count[approach] = (
            len(approach_sentences[approach]) / total_num_sentences_in_approaches * 100
        )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[approach_sentence_count["rationality"]],
            name="racionalidade",
            orientation="h",
            marker=dict(color="green"),
            hovertemplate="racionalidade: %{x:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[approach_sentence_count["intuition"]],
            name="intuição",
            orientation="h",
            marker=dict(color="red"),
            hovertemplate="intuição: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Racionalidade vs Intuição",
        barmode="stack",
        xaxis=dict(
            showgrid=False,  # thin lines in the background
            zeroline=False,  # thick line at x=0
            visible=False,  # numbers below
        ),
        yaxis=dict(
            showgrid=False,  # thin lines in the background
            zeroline=False,  # thick line at x=0
            visible=False,  # numbers below
        ),
    )
    return fig
