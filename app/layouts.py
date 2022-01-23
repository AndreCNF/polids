import streamlit as st
import os
import sys

sys.path.append("utils/")
from data_utils import DATA_DIR, load_yaml_file, load_markdown_file
from nlp_utils import get_sentences
from viz_utils import get_word_cloud

DATA_NAME = "portugal_2022"
PARTY_DATA = load_yaml_file(os.path.join(DATA_DIR, DATA_NAME, "parties_data.yml"))
PARTY_NAMES = list(PARTY_DATA.keys())
TOPICS = load_yaml_file(os.path.join(DATA_DIR, DATA_NAME, "topics.yml"))
APPROACHES = load_yaml_file(os.path.join(DATA_DIR, DATA_NAME, "approaches.yml"))


def geral():
    st.title("Análise geral de todos os partidos")
    # initial instructions
    init_info = st.empty()
    init_info.info(
        "ℹ️ Nesta página, tens uma visão geral de todos os partidos. "
        "Para veres o resultado de cada partido, abre a barra lateral (se "
        "não estiver aberta, clica no símbolo ☰ no canto superior direito) "
        "e escolha a opção 'Individual'."
    )
    # load data
    programs_txt = [
        load_markdown_file(os.path.join(DATA_DIR, DATA_NAME, "programs", f"{party}.md"))
        for party in PARTY_NAMES
    ]
    sentences = [get_sentences(txt) for txt in programs_txt]
    # flatten
    sentences = [sent for sublist in sentences for sent in sublist]
    # word cloud card
    st.subheader("Palavras mais frequentes")
    st.image(os.path.join(DATA_DIR, DATA_NAME, "word_clouds", "all_parties.png"))

    # with st.spinner("A analisar sentimentos..."):
    # with st.spinner("A analisar discurso ódio..."):


def individual():
    st.title("Análise individual de cada partido")
    # initial instructions
    init_info = st.empty()
    init_info.info(
        "ℹ️ Aqui encontras a análise de cada partido. Podes escolher qual "
        "analisar através do menu na barra lateral (se não estiver aberta, "
        "clica no símbolo ☰ no canto superior direito)."
    )
