from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel, Field


class SemanticChunksPerPage(BaseModel):
    chunks: List[str] = Field(
        description="List of semantic chunks extracted from the page, exactly matching segments of the original text."
    )
    last_chunk_incomplete: bool = Field(
        description="Boolean flag that is true if the last chunk's topic continues into the next page, false otherwise."
    )


class TextChunker(ABC):
    @abstractmethod
    def get_chunks(
        self,
        text_per_page: List[str],
        skip_similarity: bool = False,
    ) -> List[SemanticChunksPerPage]:
        """
        Splits the input text into semantic chunks.

        Args:
            text_per_page (List[str]): The input text to split, listed by page.
                Each page is a separate string in the list. The text is
                expected to be in Markdown format.
            skip_similarity (bool): If True, skip cross-page similarity checks.

        Returns:
            List[SemanticChunksPerPage]: A list of semantic chunks.
        """
        pass

    @abstractmethod
    def merge_chunks(self, chunks: List[SemanticChunksPerPage]) -> List[str]:
        """
        Merges incomplete chunks from the end of one page with the beginning of the next.

        Args:
            chunks (List[str]): A list of chunks to merge.

        Returns:
            List[str]: A list of merged chunks.
        """
        pass

    def process(
        self,
        text_per_page: List[str],
        raw_chunks_only: bool = False,
    ) -> List[str]:
        """
        Processes the input text to extract semantic chunks.

        Args:
            text_per_page (List[str]): The input text to split, listed by page.
                Each page is a separate string in the list. The text is
                expected to be in Markdown format.
            raw_chunks_only (bool): If True, skip similarity checks and chunk merging,
                returning raw per-page chunks only.

        Returns:
            List[str]: A list of semantic chunks.
        """
        # Get semantic chunks per page, optionally skipping similarity checks
        page_chunks = self.get_chunks(text_per_page, skip_similarity=raw_chunks_only)
        if raw_chunks_only:
            # Flatten raw cleaned chunks per page without merging
            return [chunk for page in page_chunks for chunk in page.chunks]
        else:
            # Merge incomplete chunks across pages
            return self.merge_chunks(page_chunks)
