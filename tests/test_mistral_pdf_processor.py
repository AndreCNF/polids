from pathlib import Path

from polids.pdf_processing.mistral import MistralPDFProcessor  # type: ignore[import]


class _Page:
    def __init__(self, markdown: str, tables: list | None = None):
        self.markdown = markdown
        self.tables = tables or []


class _Table:
    def __init__(self, table_id: str, content: str):
        self.id = table_id
        self.content = content


class _Response:
    def __init__(self, pages: list[_Page]):
        self.pages = pages


class _BatchJob:
    def __init__(
        self,
        job_id: str = "job_1",
        outputs: list | None = None,
        output_file: str | None = None,
    ):
        self.id = job_id
        self.outputs = outputs
        self.output_file = output_file


class _DownloadedFile:
    def __init__(self, text: str):
        self._raw = text.encode("utf-8")
        self._was_read = False

    @property
    def text(self):
        if not self._was_read:
            raise RuntimeError("Attempted to access streaming response content")
        return self._raw.decode("utf-8")

    def read(self):
        self._was_read = True
        return self._raw


class _FilesAPI:
    def __init__(self, text: str):
        self.text = text
        self.downloaded_file_id: str | None = None

    def download(self, *, file_id: str):
        self.downloaded_file_id = file_id
        return _DownloadedFile(self.text)


class _Client:
    def __init__(self, text: str):
        self.files = _FilesAPI(text)


def test_process_returns_markdown_pages(monkeypatch):
    processor = MistralPDFProcessor.__new__(MistralPDFProcessor)

    monkeypatch.setattr(processor, "_encode_pdf", lambda _: "encoded-pdf")
    monkeypatch.setattr(
        processor,
        "_run_ocr",
        lambda _: _Response([_Page(" page 1 "), _Page("page 2")]),
    )

    result = processor.process(Path("tests/data/test_electoral_program.pdf"))

    assert result == ["page 1", "page 2"]


def test_process_handles_empty_page_markdown(monkeypatch):
    processor = MistralPDFProcessor.__new__(MistralPDFProcessor)

    monkeypatch.setattr(processor, "_encode_pdf", lambda _: "encoded-pdf")
    monkeypatch.setattr(
        processor,
        "_run_ocr",
        lambda _: _Response([_Page(""), _Page("valid page")]),
    )

    result = processor.process(Path("tests/data/test_electoral_program.pdf"))

    assert result == ["Error processing page 1", "valid page"]


def test_process_replaces_table_references_and_removes_image_references(monkeypatch):
    processor = MistralPDFProcessor.__new__(MistralPDFProcessor)

    page = _Page(
        markdown="# Index\n\n[tbl-0.md](tbl-0.md)\n\n[img-0.jpeg](img-0.jpeg)\n",
        tables=[_Table("tbl-0.md", "| Col A | Col B |\n| --- | --- |\n| X | Y |")],
    )

    monkeypatch.setattr(processor, "_encode_pdf", lambda _: "encoded-pdf")
    monkeypatch.setattr(processor, "_run_ocr", lambda _: _Response([page]))

    result = processor.process(Path("tests/data/test_electoral_program.pdf"))

    assert result == ["# Index\n\n| Col A | Col B |\n| --- | --- |\n| X | Y |"]


def test_process_batch_returns_results_in_input_order(monkeypatch):
    processor = MistralPDFProcessor.__new__(MistralPDFProcessor)
    processor.model = "mistral-ocr-latest"

    pdf_paths = [Path("first.pdf"), Path("second.pdf")]

    monkeypatch.setattr(
        processor,
        "_upload_pdf_files_for_batch",
        lambda _paths: ["file_1", "file_2"],
    )
    monkeypatch.setattr(
        processor,
        "_create_batch_input_file",
        lambda _file_ids: "batch_input_1",
    )
    monkeypatch.setattr(
        processor,
        "_create_batch_job",
        lambda _batch_input_file_id: _BatchJob(job_id="job_123"),
    )
    monkeypatch.setattr(
        processor,
        "_wait_for_batch_job_completion",
        lambda **_: _BatchJob(
            outputs=[
                {
                    "custom_id": "1",
                    "response": {
                        "body": {"pages": [{"markdown": "second-page", "tables": []}]}
                    },
                },
                {
                    "custom_id": "0",
                    "response": {
                        "body": {"pages": [{"markdown": "first-page", "tables": []}]}
                    },
                },
            ]
        ),
    )
    monkeypatch.setattr(processor, "_extract_batch_outputs", lambda job: job.outputs)

    result = processor.process_batch(pdf_paths)

    assert result == [["first-page"], ["second-page"]]


def test_extract_batch_outputs_prefers_output_file_download():
    processor = MistralPDFProcessor.__new__(MistralPDFProcessor)
    processor.client = _Client(
        "\n".join(
            [
                '{"custom_id":"0","response":{"body":{"pages":[{"markdown":"first","tables":[]}]}}}',
                '{"custom_id":"1","response":{"body":{"pages":[{"markdown":"second","tables":[]}]}}}',
            ]
        )
    )

    outputs = processor._extract_batch_outputs(
        _BatchJob(
            outputs=[{"custom_id": "0"}],
            output_file="file_out_123",
        )
    )

    assert processor.client.files.downloaded_file_id == "file_out_123"
    assert len(outputs) == 2
