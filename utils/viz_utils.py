from typing import Dict, List
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure
import os
import pandas as pd

from data_utils import get_counts
from nlp_utils import get_topical_sentences


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
    height: int = 300,
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
        height (int):
            Height of the plot.

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
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        height=height,
    )
    fig.update_traces(marker_color=color)
    return fig


def plot_approaches(
    sentences: List[str],
    approaches: Dict[str, List[str]],
    title: str = None,
    height: int = 300,
) -> Figure:
    """
    Plot the approaches taken to language and policy.

    Args:
        sentences (List[str]):
            List of sentences to analyse.
        approaches (Dict[str, List[str]]):
            Dictionary of words per approach.
        title (str):
            Title of the plot.
        height (int):
            Height of the plot.

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
        title=title,
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
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        height=height,
    )
    return fig


def plot_sentiment(
    df: pd.DataFrame, title: str = None, height: int = 300, label_col: str = "label"
) -> Figure:
    """
    Plot the predicted sentiment of the sentences.

    Args:
        df (pd.DataFrame):
            Dataframe with the outputs of a sentiment analysis model.
        title (str):
            Title of the plot.
        height (int):
            Height of the plot.
        label_col (str):
            Column name of the sentiment.

    Returns:
        Figure:
            Plotly figure with the percentage of hate speech.
    """
    sentiments_count = get_counts(df, label_col=label_col)
    labels_order = ["neutro", "positivo", "negativo"]
    fig = px.bar(
        x=labels_order,
        y=[
            float(sentiments_count[sentiments_count[label_col] == label].percent)
            for label in labels_order
        ],
        title=title,
    )
    fig.update_traces(
        marker_color=["gray", "green", "red"],
        hovertemplate="%{y:.1f}%<extra></extra>",
    )
    fig.update_layout(
        xaxis_title="Sentimento",
        yaxis_title="Percentagem de frases",
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        height=height,
    )
    return fig


def plot_hate_speech(
    df: pd.DataFrame, title: str = None, height: int = 300, label_col: str = "label"
) -> Figure:
    """
    Show the percentage of estimated hate speech sentences.

    Args:
        df (pd.DataFrame):
            Dataframe with the outputs of a hate speech model.
        title (str):
            Title of the plot.
        height (int):
            Height of the plot.
        label_col (str):
            Column name of the hate speech.

    Returns:
        Figure:
            Plotly figure with the percentage of hate speech.
    """
    hate_count = get_counts(df, label_col=label_col)
    try:
        hate_percent = hate_count[hate_count[label_col] == "ódio"].percent.values[0]
    except IndexError:
        hate_percent = 0
    fig = go.Figure(
        go.Indicator(
            mode="number",
            value=hate_percent,
            title=title,
            number=dict(suffix="%", valueformat=".2f"),
            delta=dict(position="top", reference=320),
            domain=dict(x=[0, 1], y=[0, 1]),
        )
    )
    fig.update_layout(
        paper_bgcolor="darkred",
        font_color="white",
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        height=height,
    )
    return fig
