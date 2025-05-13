import pytest

from polids.text_chunking.markdown_chunker import MarkdownTextChunker  # type: ignore[import]
from polids.text_chunking.base import SemanticChunksPerPage  # type: ignore[import]
from polids.utils.text_similarity import is_text_similar  # type: ignore[import]


def test_get_chunks_basic_heading_split():
    """
    Test that MarkdownTextChunker splits text at Markdown headings correctly
    and sets last_chunk_incomplete=False when no continuation is detected.
    """
    pages = [
        "# Title\nIntroduction to the document.",
        "## Section\nContent of the section.",
    ]
    chunker = MarkdownTextChunker()
    chunks = chunker.get_chunks(pages)

    assert isinstance(chunks, list), f"Expected list, got {type(chunks)}."
    assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}."

    # Each page has one heading chunk
    assert isinstance(chunks[0], SemanticChunksPerPage), (
        f"First page is not SemanticChunksPerPage: {type(chunks[0])}."
    )
    assert len(chunks[0].chunks) == 1, (
        f"First page should have 1 subchunk, got {len(chunks[0].chunks)}."
    )
    assert chunks[0].chunks[0].startswith("# Title"), (
        f"First page does not start with '# Title': {chunks[0].chunks[0]}"
    )

    assert len(chunks[1].chunks) == 1, (
        f"Second page should have 1 subchunk, got {len(chunks[1].chunks)}."
    )
    assert chunks[1].chunks[0].startswith("## Section"), (
        f"Second page does not start with '## Section': {chunks[1].chunks[0]}"
    )


def test_get_chunks_continuation_without_heading():
    """
    When pages have no headings, entire text is one chunk and similarity check
    should detect continuation.
    """
    pages = ["This is part one of a sentence", "part one of a sentence continues here"]
    chunker = MarkdownTextChunker()
    chunks = chunker.get_chunks(pages)

    # Both pages produce one chunk each
    assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}."
    assert len(chunks[0].chunks) == 1, (
        f"First page should have 1 subchunk, got {len(chunks[0].chunks)}."
    )
    assert len(chunks[1].chunks) == 1, (
        f"Second page should have 1 subchunk, got {len(chunks[1].chunks)}."
    )

    # similarity-based continuation flag should be True
    assert chunks[0].last_chunk_incomplete is True, (
        f"First page last_chunk_incomplete should be True, got {chunks[0].last_chunk_incomplete}."
    )
    assert chunks[1].last_chunk_incomplete is False, (
        f"Second page last_chunk_incomplete should be False, got {chunks[1].last_chunk_incomplete}."
    )


def test_merge_chunks_manual_semantic_merge():
    """
    Test merging when SemanticChunksPerPage list is provided manually,
    combining incomplete last chunk with the next page's first chunk.
    """
    pages = [
        SemanticChunksPerPage(chunks=["A part"], last_chunk_incomplete=True),
        SemanticChunksPerPage(chunks=["A part continues"], last_chunk_incomplete=False),
        SemanticChunksPerPage(chunks=["Another page"], last_chunk_incomplete=False),
    ]
    chunker = MarkdownTextChunker()
    merged = chunker.merge_chunks(pages)

    # First two should merge, third stays separate
    assert len(merged) == 2, f"Expected 2 merged chunks, got {len(merged)}."
    assert is_text_similar("A part\n\nA part continues", merged[0]), (
        f"Merged chunk 0 not similar. Expected: 'A part\n\nA part continues', Got: {merged[0]}"
    )
    assert merged[1] == "Another page", (
        f"Merged chunk 1 should be 'Another page', got: {merged[1]}"
    )


def test_process_integration():
    """
    Integration test combining get_chunks and merge_chunks via process().
    """
    pages = ["Intro text", "Intro text continues", "Standalone text"]
    chunker = MarkdownTextChunker()
    result = chunker.process(pages)

    # The first two should merge, third remains
    assert len(result) == 2, f"Expected 2 final chunks, got {len(result)}."
    assert is_text_similar("Intro text\n\nIntro text continues", result[0]), (
        f"First merged chunk not similar. Expected: 'Intro text\n\nIntro text continues', Got: {result[0]}"
    )
    assert result[1] == "Standalone text", (
        f"Second merged chunk should be 'Standalone text', got: {result[1]}"
    )
