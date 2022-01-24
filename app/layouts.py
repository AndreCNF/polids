import streamlit as st
import os
import sys

sys.path.append("utils/")
from data_utils import DATA_DIR
from app_utils import (
    DATA_NAME,
    PARTY_DATA,
    PARTY_NAMES,
    PARTY_FULL_NAMES,
    display_main_analysis,
)


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
    # main analysis layout
    display_main_analysis("all_parties")


def individual():
    st.title("Análise individual de cada partido")
    # initial instructions
    init_info = st.empty()
    init_info.info(
        "ℹ️ Aqui encontras a análise de cada partido. Podes escolher qual "
        "analisar através do menu na barra lateral (se não estiver aberta, "
        "clica no símbolo ☰ no canto superior direito)."
    )
    # party selection
    party = st.sidebar.selectbox("Escolha o partido", PARTY_FULL_NAMES)
    party_key = [
        party_name
        for party_name in PARTY_NAMES
        if PARTY_DATA[party_name]["full_name"] == party
    ][0]
    left_col, right_col = st.columns([1, 3])
    left_col.image(os.path.join(DATA_DIR, DATA_NAME, "logos", f"{party_key}.png"))
    right_col.title(f"{party}")
    # main analysis layout
    display_main_analysis(party)
    # with st.expander("Possíveis frases de ódio"):
