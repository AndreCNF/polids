import base64
from pathlib import Path
from typing import List
from pdf2image import convert_from_path
from io import BytesIO
from pydantic import BaseModel
from loguru import logger  # type: ignore[import]

from polids.config import settings
from polids.utils.backoff import llm_backoff
from polids.pdf_processing.base import PDFProcessor

if settings.langfuse.log_to_langfuse:
    # If Langfuse is enabled, use the Langfuse OpenAI client
    from langfuse.openai import OpenAI  # type: ignore[import]
else:
    from openai import OpenAI


# Set a Pydantic model for the OpenAI API response;
# this structured output approach makes it easier to
# access the parsed text from the API response,
# without manual parsing
class ParsedPDFText(BaseModel):
    text: str


class OpenAIPDFProcessor(PDFProcessor):
    def __init__(
        self,
        temperature: float | None = None,
        seed: int | None = None,
    ):
        """
        Initializes the OpenAIPDFProcessor with an OpenAI client.

        Args:
            temperature (float | None): Sampling temperature for chat completions.
            seed (int | None): Random seed for reproducibility.
        """
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.temperature = temperature
        self.seed = seed

    def pdf_to_base64_images(
        self, pdf_path: Path, image_format: str = "PNG", dpi: int = 300
    ) -> List[str]:
        """
        Convert each page of a PDF to a base64 encoded image.

        Args:
            pdf_path (Path): Path to the PDF file.
            image_format (str): Format to save the images as (e.g., PNG, JPEG).
            dpi (int): DPI resolution for the images.

        Returns:
            List[str]: List of base64 encoded strings, one for each page.
        """
        images = convert_from_path(pdf_path, dpi=dpi)
        base64_images = []
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_images.append(img_str)
        return base64_images

    @llm_backoff
    def _call_openai_completion(
        self, image_base64: str, pdf_path: Path, page_number: int
    ) -> str | None:
        """
        Calls the OpenAI chat completion API for a single page image, with backoff applied.
        """
        # Prepare optional kwargs for temperature and seed
        parse_kwargs: dict[str, float | int] = {}
        if self.temperature is not None:
            parse_kwargs["temperature"] = self.temperature
        if self.seed is not None:
            parse_kwargs["seed"] = self.seed
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4.1-mini-2025-04-14",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Parse all of the text from this image into a Markdown format.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                                "detail": "high",  # High detail for better OCR quality
                            },
                        },
                    ],
                }
            ],
            response_format=ParsedPDFText,
            **parse_kwargs,
        )
        if completion.choices[0].message.parsed:
            return completion.choices[0].message.parsed.text.strip()
        else:
            logger.warning(
                f"Parsed result is None for page {page_number} of PDF '{pdf_path.stem}'"
            )
            return None

    def process(self, pdf_path: Path, max_pages: int | None = None) -> List[str]:
        """
        Processes a PDF file and extracts text from each page in markdown format.

        Args:
            pdf_path (Path): Path to the PDF file.
            max_pages (int): Maximum number of pages to process. If None, process all pages.
                Defaults to None.

        Returns:
            List[str]: List of markdown strings, one per page.
        """
        base64_images = self.pdf_to_base64_images(pdf_path)
        markdown_pages = []

        for i, image_base64 in enumerate(base64_images):
            if max_pages and i >= max_pages:
                break
            try:
                result = self._call_openai_completion(
                    image_base64=image_base64,
                    pdf_path=pdf_path,
                    page_number=i + 1,
                )
                if result is not None:
                    markdown_pages.append(result)
                else:
                    logger.warning(
                        f"Parsed result is None for page {i + 1} of PDF '{pdf_path.stem}'"
                    )
                    markdown_pages.append(f"Error processing page {i + 1}")
            except Exception as e:
                logger.warning(
                    f"Error processing page {i + 1} of PDF '{pdf_path.stem}': {e}"
                )
                markdown_pages.append(f"Error processing page {i + 1}")

        return markdown_pages
