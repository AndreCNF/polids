import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, List, Sequence

from loguru import logger
from mistralai import Mistral

from polids.config import settings
from polids.pdf_processing.base import PDFProcessor
from polids.utils.backoff import llm_backoff


class MistralPDFProcessor(PDFProcessor):
    _TABLE_REF_PATTERN = re.compile(r"\[(tbl-[^\]]+)\]\(\1\)")
    _IMAGE_REF_PATTERN = re.compile(r"!?\[(img-[^\]]+)\]\(\1\)")

    def __init__(self, model: str = "mistral-ocr-latest"):
        """
        Initializes the MistralPDFProcessor with a Mistral client.

        Args:
            model (str): Mistral OCR model name. Defaults to "mistral-ocr-latest".
        """
        logging.getLogger("mistralai.extra.observability.otel").setLevel(logging.ERROR)
        self.client = Mistral(api_key=settings.mistral_api_key)
        self.model = model

    def _encode_pdf(self, pdf_path: Path) -> str:
        """
        Encodes a PDF file to a base64 string.

        Args:
            pdf_path (Path): Path to the PDF file.

        Returns:
            str: Base64 encoded PDF bytes.
        """
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode("utf-8")

    @llm_backoff
    def _run_ocr(self, base64_pdf: str):
        """
        Calls Mistral OCR with backoff applied.
        """
        return self.client.ocr.process(
            model=self.model,
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_pdf}",
            },  # type: ignore[arg-type]
            include_image_base64=False,
            table_format="markdown",
        )

    def process(self, pdf_path: Path) -> List[str]:
        """
        Processes a PDF file and extracts markdown text from each page.

        Args:
            pdf_path (Path): Path to the PDF file.

        Returns:
            List[str]: List of markdown strings, one per page.
        """
        base64_pdf = self._encode_pdf(pdf_path)
        response = self._run_ocr(base64_pdf)
        return self._extract_markdown_pages(response, pdf_path)

    def process_batch(
        self,
        pdf_paths: Sequence[Path],
        poll_interval_seconds: float = 2.0,
        timeout_seconds: int = 4 * 60 * 60,
    ) -> List[List[str]]:
        """
        Processes multiple PDFs using Mistral Batch OCR and returns results in
        the same order as `pdf_paths`.

        Args:
            pdf_paths (Sequence[Path]): Paths to PDF files.
            poll_interval_seconds (float): Polling interval for batch status checks.
            timeout_seconds (int): Max wait time for batch completion.

        Returns:
            List[List[str]]: Per-PDF list of markdown pages, ordered to match input paths.
        """
        if not pdf_paths:
            return []

        uploaded_pdf_file_ids = self._upload_pdf_files_for_batch(pdf_paths)
        batch_input_file_id = self._create_batch_input_file(uploaded_pdf_file_ids)
        batch_job = self._create_batch_job(batch_input_file_id)
        completed_job = self._wait_for_batch_job_completion(
            job_id=batch_job.id,
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=timeout_seconds,
        )
        outputs = self._extract_batch_outputs(completed_job)
        results_by_index = self._parse_batch_outputs(outputs, pdf_paths)
        return [results_by_index[index] for index in range(len(pdf_paths))]

    def _upload_pdf_files_for_batch(self, pdf_paths: Sequence[Path]) -> list[str]:
        file_ids: list[str] = []
        for pdf_path in pdf_paths:
            with open(pdf_path, "rb") as pdf_file:
                uploaded = self.client.files.upload(
                    file={
                        "file_name": pdf_path.name,
                        "content": pdf_file,
                        "content_type": "application/pdf",
                    },
                    purpose="ocr",
                )
            file_ids.append(uploaded.id)
        return file_ids

    def _create_batch_input_file(self, uploaded_pdf_file_ids: Sequence[str]) -> str:
        jsonl_lines: list[str] = []
        for index, file_id in enumerate(uploaded_pdf_file_ids):
            request_row = {
                "custom_id": str(index),
                "body": {
                    "document": {
                        "type": "file",
                        "file_id": file_id,
                    },
                    "include_image_base64": False,
                    "table_format": "markdown",
                },
            }
            jsonl_lines.append(json.dumps(request_row))

        uploaded_input_file = self.client.files.upload(
            file={
                "file_name": "mistral_ocr_batch_requests.jsonl",
                "content": ("\n".join(jsonl_lines)).encode("utf-8"),
                "content_type": "application/jsonl",
            },
            purpose="batch",
        )
        return uploaded_input_file.id

    def _create_batch_job(self, batch_input_file_id: str) -> Any:
        return self.client.batch.jobs.create(
            endpoint="/v1/ocr",
            model=self.model,
            input_files=[batch_input_file_id],
        )

    def _wait_for_batch_job_completion(
        self, job_id: str, poll_interval_seconds: float, timeout_seconds: int
    ) -> Any:
        deadline = time.monotonic() + timeout_seconds
        last_status = "UNKNOWN"

        while time.monotonic() <= deadline:
            batch_job = self.client.batch.jobs.get(job_id=job_id)
            status = str(getattr(batch_job, "status", "UNKNOWN"))
            last_status = status

            if status == "SUCCESS":
                return batch_job

            if status in {"FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"}:
                errors = getattr(batch_job, "errors", [])
                raise RuntimeError(
                    f"Mistral batch OCR job {job_id} failed with status '{status}'. Errors: {errors}"
                )

            time.sleep(poll_interval_seconds)

        raise TimeoutError(
            f"Mistral batch OCR job {job_id} did not complete within {timeout_seconds}s (last status: {last_status})"
        )

    def _extract_batch_outputs(self, batch_job: Any) -> list[dict[str, Any]]:
        output_file = getattr(batch_job, "output_file", None)
        if isinstance(output_file, str) and output_file:
            response = self.client.files.download(file_id=output_file)
            return [
                json.loads(line) for line in response.text.splitlines() if line.strip()
            ]

        outputs = getattr(batch_job, "outputs", None)
        if isinstance(outputs, list) and outputs:
            return [self._to_dict(output) for output in outputs]

        raise RuntimeError(
            f"Batch OCR job {getattr(batch_job, 'id', '<unknown>')} has no inline outputs and no output_file."
        )

    def _parse_batch_outputs(
        self, outputs: Sequence[dict[str, Any]], pdf_paths: Sequence[Path]
    ) -> dict[int, List[str]]:
        results_by_index: dict[int, List[str]] = {}

        for output in outputs:
            custom_id = self._extract_custom_id(output)
            if custom_id is None:
                logger.warning(
                    f"Skipping batch output with invalid custom_id: {output.get('custom_id')}"
                )
                continue

            index = custom_id
            if not 0 <= index < len(pdf_paths):
                logger.warning(
                    f"Skipping batch output with out-of-range custom_id: {custom_id}"
                )
                continue

            error = output.get("error")
            if error:
                raise RuntimeError(
                    f"Mistral batch OCR returned an error for request {custom_id}: {error}"
                )

            response = output.get("response")
            if isinstance(response, dict):
                status_code = response.get("status_code")
                if isinstance(status_code, int) and status_code >= 400:
                    raise RuntimeError(
                        f"Mistral batch OCR request {custom_id} failed with status {status_code}: {response.get('body')}"
                    )

            body = self._extract_output_body(output)
            if not isinstance(body, dict):
                raise RuntimeError(
                    f"Mistral batch OCR output for request {custom_id} has no valid response body."
                )

            results_by_index[index] = self._extract_markdown_pages(
                body, pdf_paths[index]
            )

        missing = [str(i) for i in range(len(pdf_paths)) if i not in results_by_index]
        if missing:
            raise RuntimeError(
                f"Missing OCR batch outputs for request(s): {', '.join(missing)}"
            )

        return results_by_index

    def _extract_custom_id(self, output: dict[str, Any]) -> int | None:
        custom_id = output.get("custom_id")
        if custom_id is None and isinstance(output.get("request"), dict):
            custom_id = output["request"].get("custom_id")

        if isinstance(custom_id, int):
            return custom_id
        if isinstance(custom_id, str) and custom_id.isdigit():
            return int(custom_id)
        return None

    def _extract_output_body(self, output: dict[str, Any]) -> dict[str, Any] | None:
        response = output.get("response")
        if isinstance(response, dict):
            body = response.get("body")
            if isinstance(body, dict):
                return body
            if isinstance(body, str):
                try:
                    parsed = json.loads(body)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    return None

        body = output.get("body")
        if isinstance(body, dict):
            return body
        if isinstance(body, str):
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return None

        return None

    def _extract_markdown_pages(self, ocr_response: Any, pdf_path: Path) -> List[str]:
        pages = self._get_value(ocr_response, "pages", [])

        markdown_pages: List[str] = []
        for index, page in enumerate(pages, start=1):
            page_markdown = self._get_value(page, "markdown", "")
            page_tables = self._get_value(page, "tables", []) or []

            page_markdown = self._replace_table_references(
                page_markdown=page_markdown,
                page_tables=page_tables,
                page_number=index,
                pdf_path=pdf_path,
            )
            page_markdown = self._remove_image_references(page_markdown).strip()
            if page_markdown:
                markdown_pages.append(page_markdown)
            else:
                logger.warning(
                    f"Empty OCR output for page {index} of PDF '{pdf_path.stem}'"
                )
                markdown_pages.append(f"Error processing page {index}")

        return markdown_pages

    def _replace_table_references(
        self,
        page_markdown: str,
        page_tables: Sequence[object],
        page_number: int,
        pdf_path: Path,
    ) -> str:
        """
        Replaces Mistral table references (e.g. [tbl-0.md](tbl-0.md))
        with their corresponding markdown table content from `page.tables`.
        """
        table_map: dict[str, str] = {}
        for table in page_tables:
            table_id = self._get_value(table, "id", None)
            table_content = self._get_value(table, "content", None)
            if isinstance(table_id, str) and isinstance(table_content, str):
                table_map[table_id] = table_content

        def _replace(match: re.Match[str]) -> str:
            table_id = match.group(1)
            if table_id in table_map:
                return table_map[table_id]
            logger.warning(
                f"Missing table content for reference '{table_id}' on page {page_number} of PDF '{pdf_path.stem}'"
            )
            return ""

        return self._TABLE_REF_PATTERN.sub(_replace, page_markdown)

    def _remove_image_references(self, page_markdown: str) -> str:
        """
        Removes Mistral image references (e.g. [img-0.jpeg](img-0.jpeg)) from markdown.
        """
        return self._IMAGE_REF_PATTERN.sub("", page_markdown)

    def _get_value(self, obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _to_dict(self, obj: Any) -> dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        raise TypeError(f"Unsupported output type for batch parsing: {type(obj)}")
