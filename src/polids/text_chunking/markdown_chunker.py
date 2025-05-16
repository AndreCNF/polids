import re
from loguru import logger
from .base import TextChunker, SemanticChunksPerPage
from ..utils.text_similarity import is_text_similar

# Precompile regex patterns for efficiency
_HEADING_REGEX = re.compile(r"(?m)(?=^#{1,6}\s)")


class MarkdownTextChunker(TextChunker):
    """
    TextChunker implementation that splits text into semantic chunks based on Markdown syntax.
    """

    def get_chunks(
        self,
        text_per_page: list[str],
        skip_similarity: bool = False,
    ) -> list[SemanticChunksPerPage]:
        """
        Splits each page's Markdown text into semantic chunks where each chunk starts with a Markdown heading.

        Args:
            text_per_page (list[str]): List of page texts in Markdown format.
            skip_similarity (bool): If True, skip cross-page similarity checks.

        Returns:
            list[SemanticChunksPerPage]: List of semantic chunks per page with incomplete flags.
        """
        page_chunks: list[SemanticChunksPerPage] = []
        for page_index, page_text in enumerate(text_per_page):
            logger.info(
                f"Splitting page {page_index + 1} into chunks based on Markdown headings."
            )
            # Split raw text at Markdown headings (H1-H6)
            raw_chunks = _HEADING_REGEX.split(page_text)
            cleaned_chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip()]
            # If skipping similarity, add raw cleaned chunks only
            if skip_similarity:
                page_chunks.append(
                    SemanticChunksPerPage(
                        chunks=cleaned_chunks,
                        last_chunk_incomplete=False,
                    )
                )
                continue
            # Determine if the last chunk continues on the next page via text similarity
            if cleaned_chunks and page_index < len(text_per_page) - 1:
                # Prepare the next page's first chunk
                next_raw = _HEADING_REGEX.split(text_per_page[page_index + 1])
                next_cleaned = [c.strip() for c in next_raw if c.strip()]
                next_first = next_cleaned[0] if next_cleaned else ""
                last_chunk = cleaned_chunks[-1]
                last_chunk_incomplete = is_text_similar(last_chunk, next_first)
                logger.debug(
                    f"Page {page_index + 1}: last chunk similarity to next first chunk = {last_chunk_incomplete}"
                )
            else:
                last_chunk_incomplete = False
            logger.debug(
                f"Page {page_index + 1}: found {len(cleaned_chunks)} chunks, "
                f"last_chunk_incomplete={last_chunk_incomplete}"
            )
            page_chunks.append(
                SemanticChunksPerPage(
                    chunks=cleaned_chunks,
                    last_chunk_incomplete=last_chunk_incomplete,
                )
            )
        logger.success("Completed splitting all pages into semantic chunks.")
        return page_chunks

    def merge_chunks(self, chunks: list[SemanticChunksPerPage]) -> list[str]:
        """
        Merges incomplete chunks across pages to form a final list of semantic chunks.

        Args:
            chunks (list[SemanticChunksPerPage]): List of chunks per page.

        Returns:
            list[str]: List of merged semantic chunks.
        """
        merged: list[str] = []
        pending: str = ""
        for _, page_chunk in enumerate(chunks):
            for idx, chunk in enumerate(page_chunk.chunks):
                # Prepend any pending chunk from previous page to the first chunk
                if idx == 0 and pending:
                    chunk = pending + "\n\n" + chunk
                    pending = ""
                is_last_chunk = idx == len(page_chunk.chunks) - 1
                if is_last_chunk and page_chunk.last_chunk_incomplete:
                    pending = chunk
                else:
                    merged.append(chunk)
        # Append any remaining pending chunk after all pages are processed
        if pending:
            merged.append(pending)
        logger.success(
            f"Completed merging all semantic chunks across pages. Total: {len(merged)}"
        )
        return merged

    def process(
        self,
        text_per_page: list[str],
        raw_chunks_only: bool = False,
    ) -> list[str]:
        """
        Processes the input text to extract semantic chunks.

        Args:
            text_per_page (list[str]): The input text to split, listed by page.
                Each page is a separate string in the list. The text is
                expected to be in Markdown format.
            raw_chunks_only (bool): If True, skip similarity checks and chunk merging,
                returning raw per-page chunks only.

        Returns:
            list[str]: A list of semantic chunks.
        """
        # Get semantic chunks per page, optionally skipping similarity checks
        page_chunks = self.get_chunks(text_per_page, skip_similarity=raw_chunks_only)
        if raw_chunks_only:
            # Flatten raw cleaned chunks per page without merging
            return [chunk for page in page_chunks for chunk in page.chunks]
        else:
            # Merge incomplete chunks across pages
            return self.merge_chunks(page_chunks)
