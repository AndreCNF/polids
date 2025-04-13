import os
import re
from pathlib import Path
from typing import List
from polids.pdf_processing.base import PDFProcessor
from marker.converters.pdf import PdfConverter  # type: ignore[import]
from marker.models import create_model_dict  # type: ignore[import]
from marker.output import text_from_rendered  # type: ignore[import]
from marker.config.parser import ConfigParser  # type: ignore[import]


class MarkerPDFProcessor(PDFProcessor):
    def __init__(self, use_llm: bool = True):
        """
        Initializes the MarkerPDFProcessor.

        Args:
            use_llm (bool): Whether to use LLM for improving output quality. Defaults to True.
        """
        self.config = {
            "use_llm": use_llm,
            "gemini_api_key": os.environ.get("GOOGLE_API_KEY"),
            "paginate_output": True,  # Separate pages in the markdown, using 48x "-" separator
            "output_format": "markdown",
        }

    def process(self, pdf_path: Path) -> List[str]:
        """
        Processes a PDF file and returns its content in markdown format as a list of strings, one per page.

        Args:
            pdf_path (Path): Path to the PDF file.

        Returns:
            List[str]: List of strings in markdown format, where each string represents the content of a page.
        """
        config_parser = ConfigParser(self.config)
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )
        rendered = converter(str(pdf_path))
        text, _, _ = text_from_rendered(rendered)
        # Remove {page_number} from separators
        cleaned_text = re.sub(r"\{\d+\}(-{48})", r"\1", text)
        # Remove image references
        cleaned_text = re.sub(r"!\[.*?\]\([^\)]+\)", "", cleaned_text)
        # Split the cleaned text into pages
        text_split_by_pages = cleaned_text.split(48 * "-")[1:]
        # Trim leading and trailing whitespace from each page
        return [page.strip() for page in text_split_by_pages]
