from typing import List, Union, Tuple
import streamlit as st
import os
import pandas as pd

from data_utils import DATA_DIR, load_markdown_file, load_yaml_file
from nlp_utils import get_sentences
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
NLP_DF = pd.read_csv(os.path.join(DATA_DIR, DATA_NAME, "nlp_outputs.csv"), sep=";")
NLP_DF.drop(columns=["Unnamed: 0"], inplace=True)


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
        program_txt = load_markdown_file(
            os.path.join(DATA_DIR, DATA_NAME, "programs", f"{party}.md")
        )
        sentences = get_sentences(program_txt)
        return sentences, program_txt


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
    if party == "all_parties":
        df = NLP_DF
    else:
        df = NLP_DF[NLP_DF["party"] == party]
    sentences = list(df["sentence"])
    st.subheader("Palavras mais frequentes")
    st.image(os.path.join(DATA_DIR, DATA_NAME, "word_clouds", f"{party}.png"))
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
            plot_sentiment(df, height=300, label_col="sentiment_label"),
            use_container_width=True,
        )
    with right_col:
        st.subheader("Racionalidade vs Intuição")
        st.plotly_chart(
            plot_approaches(sentences, APPROACHES, height=40), use_container_width=True
        )
        st.subheader("Percentagem estimada de discurso de ódio")
        st.plotly_chart(
            plot_hate_speech(df, height=100, label_col="hate_speech_label"),
            use_container_width=True,
        )
    if party != "all_parties":
        sentences, program_txt = get_sentences_from_party(party)
        return df, program_txt
