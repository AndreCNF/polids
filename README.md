<div align="center">

![polids](https://github.com/AndreCNF/polids/blob/main/data/polids_logo.png?raw=true)

[![Code style: ruff](https://camo.githubusercontent.com/bb88127790fb054cba2caf3f3be2569c1b97bb45a44b47b52d738f8781a8ede4/68747470733a2f2f696d672e736869656c64732e696f2f656e64706f696e743f75726c3d68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f636861726c6965726d617273682f727566662f6d61696e2f6173736574732f62616467652f76312e6a736f6e)](https://github.com/charliermarsh/ruff)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/andrecnf/polids/main/app/app.py)

Analysis of political data and output of it through apps.

</div>

## Work in progress

This repo is currently undergoing refactoring, so as to become more easily applicable to elections in any time, in any language.

## Setup
### Installation

To install the package and its dependencies, use [uv](https://docs.astral.sh/uv/) and:

1. Install dependencies:

   Just the basics:
   ```bash
   uv sync
   ```

   Including dev dependencies:
   ```bash
   uv sync --all-extras --dev
   ```

1. Install the package:
   ```bash
   uv pip install -e .
   ```

### Setting up Pre-commit Hooks

To ensure code is formatted correctly before committing, set up pre-commit hooks:

1. Setup `pre-commit` by installing the dev dependencies:
   ```bash
   uv sync --dev
   ```

2. Install the hooks:
   ```bash
   uv run pre-commit install
   ```

3. Run the hooks manually on all files (optional):
   ```bash
   uv run pre-commit run --all-files
   ```

This will use `ruff` to format your code automatically.
