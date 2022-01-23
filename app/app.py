import streamlit as st
import layouts

PAGES = dict(Geral=layouts.geral, Individual=layouts.individual)

st.sidebar.title("Polids | 🇵🇹 2022")
st.sidebar.markdown("Análise de dados dos programas políticos a eleições.")
st.sidebar.markdown(
    "[Mais info](https://github.com/AndreCNF/elections-analysis)",
    unsafe_allow_html=True,
)
st.sidebar.title("Navegação")
selection = st.sidebar.radio("Ir para", list(PAGES.keys()))
page = PAGES[selection]
page()
