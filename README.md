<div align="center">

![polids](https://github.com/AndreCNF/polids/blob/main/data/polids_logo.png?raw=true)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/andrecnf/polids/main/app/app.py)

Analysis of political data and output of it through apps.

</div>

## Work in progress

This repo is currently undergoing refactoring, so as to become more easily applicable to elections in any time, in any language.

## Setting up Pre-commit Hooks

To ensure code is formatted correctly before committing, set up pre-commit hooks:

1. Install `pre-commit`:
   ```bash
   pip install pre-commit
   ```

2. Install the hooks:
   ```bash
   pre-commit install
   ```

3. Run the hooks manually on all files (optional):
   ```bash
   pre-commit run --all-files
   ```

This will use `ruff` to format your code automatically.
