from typing import List, Union, Tuple
import streamlit as st
import os
import pandas as pd

from data_utils import DATA_DIR, load_markdown_file, load_yaml_file
from nlp_utils import get_sentences, get_sentiment, get_hate_speech
from viz_utils import (
    plot_topical_presence,
    plot_approaches,
    plot_sentiment,
    plot_hate_speech,
)

DATA_NAME = "portugal_2022"
PARTY_DATA = load_yaml_file(os.path.join(DATA_DIR, DATA_NAME, "parties_data.yml"))
PARTY_NAMES = list(PARTY_DATA.keys())
PARTY_FULL_NAMES = [PARTY_DATA[party]["full_name"] for party in PARTY_NAMES]
TOPICS = load_yaml_file(os.path.join(DATA_DIR, DATA_NAME, "topics.yml"))
APPROACHES = load_yaml_file(os.path.join(DATA_DIR, DATA_NAME, "approaches.yml"))


def get_sentences_from_party(party: str) -> Tuple[List[str], Union[None, str]]:
    """
    Get the list of sentences for a party.

    Args:
        party (str):
            Party to get the list of sentences for.

    Returns:
        List[str]:
            List of sentences for the party.
        str:
            Party program.
    """
    if party == "all_parties":
        party_files = os.listdir(os.path.join(DATA_DIR, DATA_NAME, "programs"))
        programs_txt = [
            load_markdown_file(os.path.join(DATA_DIR, DATA_NAME, "programs", party_f))
            for party_f in party_files
        ]
        sentences = [get_sentences(txt) for txt in programs_txt]
        # flatten
        sentences = [sent for sublist in sentences for sent in sublist]
        return sentences, None
    else:
        st.write(os.path.join(DATA_DIR, DATA_NAME, "programs", f"{party}.md"))
        st.write(os.getcwd())
        program_txt = load_markdown_file(
            os.path.join(DATA_DIR, DATA_NAME, "programs", f"{party}.md")
        )
        sentences = get_sentences(program_txt)
        return sentences, program_txt


@st.experimental_memo
def get_sentiment_df(party: str, data_name: str = "portugal_2022") -> pd.DataFrame:
    """
    Get the sentiment dataframe for a party and cache it.

    Args:
        party (str):
            Party to get the sentiment dataframe for. Can also be "all_parties".
        data_name (str):
            Name of the data to get the sentiment dataframe for.

    Returns:
        sentiment_df:
            Dataframe with the sentiment data.
    """
    sentences, _ = get_sentences_from_party(party)
    return get_sentiment(sentences)


@st.experimental_memo
def get_hate_speech_df(
    party: str, data_name: str, sentiment_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Get the hate speech dataframe for a party and cache it.

    Args:
        party (str):
            Party to get the hate speech dataframe for. Can also be "all_parties".
        data_name (str):
            Name of the data to get the hate speech dataframe for.
        sentiment_df (pd.DataFrame):
            Dataframe with the sentiment data.

    Returns:
        hate_speech_df:
            Dataframe with the hate speech data.
    """
    sentences, _ = get_sentences_from_party(party)
    return get_hate_speech(sentences, sentiment_df)


def display_main_analysis(party: str) -> Union[None, Tuple[pd.DataFrame, str]]:
    """
    Display the main analysis for a party.

    Args:
        party (str):
            Party to display the main analysis for.

    Returns:
        pd.DataFrame:
            Dataframe with the hate speech outputs.
            Only returned if the party is not "all_parties".
        str:
            Party program.
    """
    sentences, program_txt = get_sentences_from_party(party)
    st.subheader("Palavras mais frequentes")
    st.image(os.path.join(DATA_DIR, DATA_NAME, "word_clouds", f"{party}.png"))
    with st.spinner("A analisar sentimentos..."):
        sentiment_df = get_sentiment_df(party=party, data_name=DATA_NAME)
    with st.spinner("A analisar discurso de ódio..."):
        hate_df = get_hate_speech_df(
            party=party, data_name=DATA_NAME, sentiment_df=sentiment_df
        )
    st.subheader("Tópicos nos programas")
    st.plotly_chart(
        plot_topical_presence(
            sentences,
            TOPICS,
            color=PARTY_DATA[party]["color"] if party != "all_parties" else "gray",
            height=300,
        ),
        use_container_width=True,
    )
    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Análise de sentimentos")
        st.plotly_chart(
            plot_sentiment(sentiment_df, height=300), use_container_width=True
        )
    with right_col:
        st.subheader("Racionalidade vs Intuição")
        st.plotly_chart(
            plot_approaches(sentences, APPROACHES, height=40), use_container_width=True
        )
        st.subheader("Percentagem estimada de discurso de ódio")
        st.plotly_chart(plot_hate_speech(hate_df, height=100), use_container_width=True)
    if party != "all_parties":
        return hate_df, program_txt
