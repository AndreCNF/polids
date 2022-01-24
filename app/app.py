import streamlit as st
import os
import sys

import layouts

sys.path.append("utils/")
from data_utils import DATA_DIR

PAGES = dict(Geral=layouts.geral, Individual=layouts.individual)

st.sidebar.image(os.path.join(DATA_DIR, "polids_logo.png"))
st.sidebar.title("🇵🇹 2022")
st.sidebar.markdown("Análise de dados dos programas políticos a eleições.")
st.sidebar.markdown(
    "[Mais info](https://github.com/AndreCNF/polids)",
    unsafe_allow_html=True,
)
st.sidebar.title("Navegação")
selection = st.sidebar.radio("Ir para", list(PAGES.keys()))
page = PAGES[selection]
page()
