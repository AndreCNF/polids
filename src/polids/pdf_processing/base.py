from abc import ABC, abstractmethod
from pathlib import Path
from typing import List


class PDFProcessor(ABC):
    @abstractmethod
    def process(self, pdf_path: Path) -> List[str]:
        """
        Processes a PDF file and returns a list of strings, where each string represents
        the content of a page in the PDF.

        Args:
            pdf_path (Path): Path to the PDF file.

        Returns:
            List[str]: List of strings, one per page.
        """
        pass
