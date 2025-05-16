<div align="center">

![polids](https://github.com/AndreCNF/polids/blob/main/data/polids_logo.png?raw=true)

[![Code style: ruff](https://camo.githubusercontent.com/bb88127790fb054cba2caf3f3be2569c1b97bb45a44b47b52d738f8781a8ede4/68747470733a2f2f696d672e736869656c64732e696f2f656e64706f696e743f75726c3d68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f636861726c6965726d617273682f727566662f6d61696e2f6173736574732f62616467652f76312e6a736f6e)](https://github.com/charliermarsh/ruff)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/andrecnf/polids/main/app/app.py)

Analysis of political data and output of it through apps.

</div>

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

## Modules

- **pdf_processing**: Convert PDFs to markdown or raw text using OpenAI and fallback processors.
- **text_chunking**: Split document text into semantic chunks with OpenAI and Markdown-based chunkers.
- **party_name_extraction**: Extract full and short party names from text chunks using OpenAI.
- **structured_analysis**: Analyze chunks for policy proposals, sentiment, topic, hate speech, and political compass scores.
- **scientific_validation**: Validate extracted policy proposals with Perplexity and OpenAI scientific validators.
- **topic_unification**: Unify and map topics across multiple documents using OpenAI unification models.
- **word_cloud**: Generate word clouds to visualize term frequencies for each political party.
- **utils**: Helper functions for pandas DataFrame conversion and data manipulation.
- **config**: Centralized configuration settings for models and processing parameters.

## Pipeline Script

The entrypoint for the end-to-end processing pipeline is:

```bash
src/polids/pipeline_runner.py
```

It exposes a `process_pdfs(input_folder: str)` function that:
1. Parses PDFs into pages
2. Chunks text into semantic segments
3. Extracts party names
4. Performs structured chunk analysis
5. Validates policy proposals scientifically
6. Unifies topics across all processed documents

## Experimentation Notebooks

Interactive Jupyter notebooks are available under the `notebooks/` directory to explore each pipeline step:

- `01_pdf2markdown.ipynb` — Parse PDFs into markdown pages
- `02_text2chunks.ipynb` — Chunk text into semantic segments
- `03_party_name_extraction.ipynb` — Extract party names from chunks
- `04_complete_chunk_analysis.ipynb` — Perform structured analysis of chunks
- `05_scientific_validation_policies.ipynb` — Scientifically validate policy proposals
- `06_topic_unification.ipynb` — Unify topics across documents
- `07_word_cloud.ipynb` — Generate word clouds for term frequencies
- `08_pydantic2pandas.ipynb` — Convert Pydantic models to pandas DataFrames
- `09_data_viz.ipynb` — Create interactive data visualizations with plotly

## Unit Testing

This project uses `pytest` for testing, with tests located in the `tests/` directory mirroring the `src/polids` module structure. Key practices:

- **Focused Tests**: One behavior per test function (`test_module_functionality.py`).
- **Descriptive Naming**: Tests named to reflect their purpose (e.g., `test_text_chunker_handles_empty_input`).
- **Isolation**: Each test is independent and does not rely on shared state.
- **Fixtures & Parametrize**: Reusable setup in `conftest.py` and `@pytest.mark.parametrize` to cover multiple cases.

Run the full test suite with:

```bash
uv run pytest
```

## Roadmap
- [ ] Implement a cheaper and faster scientific validation method.
- [ ] Improve the policy proposal extraction process to only extract actionable, concrete proposals.
- [ ] Improve topic unification to avoid redundant topics.
- [ ] Speed up the pipeline by parallelizing the processing of documents and exploring faster LLM inference options.
- [ ] Create a web app and a process to run the pipeline on a server, allowing users to upload documents and receive results when ready.