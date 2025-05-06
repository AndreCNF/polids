import pytest
from typing import List
from polids.text_chunking.openai import OpenAITextChunker  # type: ignore[import]
from polids.utils import is_text_similar  # type: ignore[import]
from polids.text_chunking.base import SemanticChunksPerPage  # type: ignore[import]


@pytest.fixture
def sample_markdown_pages():
    return [
        "# Page 1\n\nThis is the first page of the document.",
        "# Page 2\n\nThis is the second page of the document. It introduces a topic that continues onto the next page.",
        "# Page 3\n\nContinuing from the previous page, this page elaborates on the topic introduced earlier.",
        "This is a standalone page with its own content.",
        "This page starts a new topic that will be expanded on the next page.",
        "Building on the topic from the previous page, this page provides final details.",
        "# Página 1\n\nEsta é a primeira página de um novo documento.",  # Portuguese
        "# Página 2\n\nEsta é a segunda página do documento. Introduz um tópico que continua na próxima página.",  # Portuguese
        "# Página 3\n\nContinuando da página anterior, esta página elabora sobre o tópico introduzido anteriormente.",  # Portuguese
        "Esta é uma página independente com seu próprio conteúdo.",  # Portuguese
        "# Página 1\n\nEsta es la primera página de un nuevo documento.",  # Spanish
        "# Página 2\n\nEsta es la segunda página del documento. Introduce un tema que continúa en la siguiente página.",  # Spanish
        "# Página 3\n\nContinuando desde la página anterior, esta página elabora sobre el tema introducido anteriormente.",  # Spanish
        "Esta es una página independiente con su propio contenido.",  # Spanish
    ]


@pytest.fixture
def expected_chunks():
    return [
        SemanticChunksPerPage(
            chunks=["# Page 1\n\nThis is the first page of the document."],
            last_chunk_incomplete=False,
        ),
        SemanticChunksPerPage(
            chunks=[
                "# Page 2\n\nThis is the second page of the document. It introduces a topic that continues onto the next page."
            ],
            last_chunk_incomplete=True,
        ),
        SemanticChunksPerPage(
            chunks=[
                "# Page 3\n\nContinuing from the previous page, this page elaborates on the topic introduced earlier."
            ],
            last_chunk_incomplete=False,
        ),
        SemanticChunksPerPage(
            chunks=["This is a standalone page with its own content."],
            last_chunk_incomplete=False,
        ),
        SemanticChunksPerPage(
            chunks=[
                "This page starts a new topic that will be expanded on the next page."
            ],
            last_chunk_incomplete=True,
        ),
        SemanticChunksPerPage(
            chunks=[
                "Building on the topic from the previous page, this page provides final details."
            ],
            last_chunk_incomplete=False,
        ),
        SemanticChunksPerPage(
            chunks=["# Página 1\n\nEsta é a primeira página de um novo documento."],
            last_chunk_incomplete=False,
        ),
        SemanticChunksPerPage(
            chunks=[
                "# Página 2\n\nEsta é a segunda página do documento. Introduz um tópico que continua na próxima página."
            ],
            last_chunk_incomplete=True,
        ),
        SemanticChunksPerPage(
            chunks=[
                "# Página 3\n\nContinuando da página anterior, esta página elabora sobre o tópico introduzido anteriormente."
            ],
            last_chunk_incomplete=False,
        ),
        SemanticChunksPerPage(
            chunks=["Esta é uma página independente com seu próprio conteúdo."],
            last_chunk_incomplete=False,
        ),
        SemanticChunksPerPage(
            chunks=["# Página 1\n\nEsta es la primera página de un nuevo documento."],
            last_chunk_incomplete=False,
        ),
        SemanticChunksPerPage(
            chunks=[
                "# Página 2\n\nEsta es la segunda página del documento. Introduce un tema que continúa en la siguiente página."
            ],
            last_chunk_incomplete=True,
        ),
        SemanticChunksPerPage(
            chunks=[
                "# Página 3\n\nContinuando desde la página anterior, esta página elabora sobre el tema introducido anteriormente."
            ],
            last_chunk_incomplete=False,
        ),
        SemanticChunksPerPage(
            chunks=["Esta es una página independiente con su propio contenido."],
            last_chunk_incomplete=False,
        ),
    ]


@pytest.fixture
def expected_merged_chunks_text():
    return [
        "# Page 1\n\nThis is the first page of the document.",
        "# Page 2\n\nThis is the second page of the document. It introduces a topic that continues onto the next page.\n\n# Page 3\n\nContinuing from the previous page, this page elaborates on the topic introduced earlier.",
        "This is a standalone page with its own content.",
        "This page starts a new topic that will be expanded on the next page.\n\nBuilding on the topic from the previous page, this page provides final details.",
        "# Página 1\n\nEsta é a primeira página de um novo documento.",
        "# Página 2\n\nEsta é a segunda página do documento. Introduz um tópico que continua na próxima página.\n\n# Página 3\n\nContinuando da página anterior, esta página elabora sobre o tópico introduzido anteriormente.",
        "Esta é uma página independente com seu próprio conteúdo.",
        "# Página 1\n\nEsta es la primera página de un nuevo documento.",
        "# Página 2\n\nEsta es la segunda página del documento. Introduce un tema que continúa en la siguiente página.\n\n# Página 3\n\nContinuando desde la página anterior, esta página elabora sobre el tema introducido anteriormente.",
        "Esta es una página independiente con su propio contenido.",
    ]


def test_get_chunks(
    sample_markdown_pages: List[str],
    expected_chunks: List[SemanticChunksPerPage],
) -> None:
    chunker = OpenAITextChunker()
    chunks = chunker.get_chunks(sample_markdown_pages)

    assert len(chunks) == len(expected_chunks), (
        f"Number of chunks {len(chunks)} does not match expected {len(expected_chunks)}."
    )
    for i, chunk in enumerate(chunks):
        assert is_text_similar(expected_chunks[i].chunks[0], chunk.chunks[0]), (
            f"Chunk {i} does not match expected.\n"
            f"Expected: {expected_chunks[i].chunks[0]}\n"
            f"Got: {chunk.chunks[0]}"
        )
        assert (
            expected_chunks[i].last_chunk_incomplete == chunk.last_chunk_incomplete
        ), (
            f"Last chunk incomplete flag mismatch for chunk {i}.\n"
            f"Expected: {expected_chunks[i].last_chunk_incomplete}\n"
            f"Got: {chunk.last_chunk_incomplete}"
        )


def test_merge_chunks(
    expected_chunks: List[SemanticChunksPerPage], expected_merged_chunks_text: List[str]
) -> None:
    chunker = OpenAITextChunker()
    merged_chunks = chunker.merge_chunks(expected_chunks)
    number_of_merged_chunks = len(merged_chunks)
    number_of_expected_merged_chunks = len(
        [chunk for chunk in expected_chunks if not chunk.last_chunk_incomplete]
    )

    assert number_of_merged_chunks == number_of_expected_merged_chunks, (
        f"Merged chunks length does not match expected.\n"
        f"Expected: {number_of_expected_merged_chunks}\n"
        f"Got: {number_of_merged_chunks}"
    )
    for i, chunk in enumerate(merged_chunks):
        assert is_text_similar(expected_merged_chunks_text[i], chunk), (
            f"Final chunk {i} does not match input page.\n"
            f"Expected: {expected_merged_chunks_text[i]}\n"
            f"Got: {chunk}"
        )


def test_process(
    sample_markdown_pages: List[str],
    expected_chunks: List[SemanticChunksPerPage],
    expected_merged_chunks_text: List[str],
) -> None:
    chunker = OpenAITextChunker()
    final_chunks = chunker.process(sample_markdown_pages)
    number_of_final_chunks = len(final_chunks)
    number_of_expected_merged_chunks = len(
        [chunk for chunk in expected_chunks if not chunk.last_chunk_incomplete]
    )

    assert number_of_final_chunks == number_of_expected_merged_chunks, (
        f"Final chunks length does not match input pages.\n"
        f"Expected: {number_of_expected_merged_chunks}\n"
        f"Got: {number_of_final_chunks}"
    )
    for i, chunk in enumerate(final_chunks):
        assert is_text_similar(expected_merged_chunks_text[i], chunk), (
            f"Final chunk {i} does not match input page.\n"
            f"Expected: {expected_merged_chunks_text[i]}\n"
            f"Got: {chunk}"
        )
