import streamlit as st
import sys

sys.path.append("../data/")


def geral():
    st.title("Análise geral de todos os partidos")
    # Initial instructions
    init_info = st.empty()
    init_info.info(
        "ℹ️ Nesta página, tens uma visão geral de todos os partidos. "
        "Para veres o resultado de cada partido, abre a barra lateral (se "
        "não estiver aberta, clica no símbolo ☰ no canto superior direito) "
        "e escolha a opção 'Individual'."
    )


def individual():
    st.title("Análise individual de cada partido")
    # Initial instructions
    init_info = st.empty()
    init_info.info(
        "ℹ️ Aqui encontras a análise de cada partido. Podes escolher qual "
        "analisar através do menu na barra lateral (se não estiver aberta, "
        "clica no símbolo ☰ no canto superior direito)."
    )
