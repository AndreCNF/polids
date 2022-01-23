import streamlit as st
import os
import pandas as pd

from data_utils import DATA_DIR, load_markdown_file
from nlp_utils import get_sentences, get_sentiment, get_hate_speech


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
    if party == "all_parties":
        party_names = os.listdir(os.path.join(DATA_DIR, data_name, "programs"))
        programs_txt = [
            load_markdown_file(
                os.path.join(DATA_DIR, data_name, "programs", f"{party}.md")
            )
            for party in party_names
        ]
        sentences = [get_sentences(txt) for txt in programs_txt]
        # flatten
        sentences = [sent for sublist in sentences for sent in sublist]
    else:
        program_txt = load_markdown_file(
            os.path.join(DATA_DIR, data_name, "programs", f"{party}.md")
        )
        sentences = get_sentences(program_txt)
    return get_sentiment(sentences)


@st.experimental_memo
def get_hate_speech_df(party: str, data_name: str = "portugal_2022") -> pd.DataFrame:
    """
    Get the hate speech dataframe for a party and cache it.

    Args:
        party (str):
            Party to get the hate speech dataframe for. Can also be "all_parties".
        data_name (str):
            Name of the data to get the hate speech dataframe for.

    Returns:
        hate_speech_df:
            Dataframe with the hate speech data.
    """
    if party == "all_parties":
        party_names = os.listdir(os.path.join(DATA_DIR, data_name, "programs"))
        programs_txt = [
            load_markdown_file(
                os.path.join(DATA_DIR, data_name, "programs", f"{party}.md")
            )
            for party in party_names
        ]
        sentences = [get_sentences(txt) for txt in programs_txt]
        # flatten
        sentences = [sent for sublist in sentences for sent in sublist]
    else:
        program_txt = load_markdown_file(
            os.path.join(DATA_DIR, data_name, "programs", f"{party}.md")
        )
        sentences = get_sentences(program_txt)
    return get_hate_speech(sentences)
