import ast
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, TypeVar

import pandas as pd  # type: ignore[import]
from loguru import logger
from tqdm.auto import tqdm  # type: ignore[import]

from polids.config import settings
from polids.party_name_extraction.openai import OpenAIPartyNameExtractor
from polids.pdf_processing.marker import MarkerPDFProcessor
from polids.pdf_processing.mistral import MistralPDFProcessor
from polids.pdf_processing.openai import OpenAIPDFProcessor
from polids.scientific_validation.gemini import GeminiScientificValidator
from polids.structured_analysis.base import (
    HateSpeechDetection,
    ManifestoChunkAnalysis,
    PoliticalCompass,
)
from polids.structured_analysis.openai import OpenAIStructuredChunkAnalyzer
from polids.text_chunking.markdown_chunker import MarkdownTextChunker
from polids.text_chunking.openai import OpenAITextChunker
from polids.topic_unification.openai import OpenAITopicUnifier
from polids.utils.pandas import convert_pydantic_to_dataframe, expand_dict_columns

TResult = TypeVar("TResult")
RATE_LIMIT_MARKERS = (
    # OpenAI error guides
    "rate limit reached for",
    "rate limit reached for requests",
    "you exceeded your current quota, please check your plan and billing details",
    "requests per min",
    "tokens per min",
    "ratelimiterror",
    "rate_limit_exceeded",
    "insufficient_quota",
    "slow down",
    # Gemini troubleshooting + API responses
    "resource_exhausted",
    "resource exhausted",
    "you've exceeded the rate limit",
    "resource has been exhausted (e.g. check quota)",
    # PydanticAI ModelHTTPError string formatting
    "status_code: 429",
    # Generic safety-net signals
    "too many requests",
    "status code: 429",
    'code":429',
    "code': 429",
    "quota exceeded",
)
ANALYSIS_TABLE_COLUMNS = [
    "pdf_file",
    "chunk_index",
    "policy_proposals",
    "sentiment",
    "topic",
    "hate_speech_is_hate_speech",
    "hate_speech_reason",
    "hate_speech_targeted_groups",
    "political_compass_economic",
    "political_compass_social",
]


def process_pdfs(input_folder: str) -> None:
    """
    Process PDFs through the full analysis pipeline and persist CSV outputs.

    The runner supports resumable execution by reusing existing CSV outputs and
    only appending missing rows for each pipeline stage.

    Runtime concurrency and retry controls are loaded from
    :class:`polids.config.Settings`.

    Args:
        input_folder (str): Path to the folder containing PDF files.
    """
    normalized_input_folder = (
        input_folder.split("=", 1)[1]
        if input_folder.startswith("input_folder=")
        else input_folder
    )
    if normalized_input_folder != input_folder:
        logger.warning(
            f"Received input in 'input_folder=...' format. Using parsed path: {normalized_input_folder}"
        )

    input_path = Path(normalized_input_folder).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input folder does not exist: {input_path}. "
            "Use a valid directory path, for example: data/elections_amsterdam/2026"
        )
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_path}")

    output_folder = input_path / "output"
    output_folder.mkdir(parents=True, exist_ok=True)

    def is_rate_limit_error(exc: Exception) -> bool:
        """Detect whether an exception was likely caused by rate limiting.

        Args:
            exc (Exception): Exception raised by an API call.

        Returns:
            bool: True when the exception likely represents a rate-limit error.
        """
        # Native SDK/PydanticAI exceptions often expose HTTP status directly.
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return True
        if status_code == 503 and "slow down" in str(exc).lower():
            return True

        # Some SDKs store status on response objects.
        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)
        if response_status == 429:
            return True

        message_parts = [str(exc)]
        body = getattr(exc, "body", None)
        if body is not None:
            message_parts.append(str(body))
        if response is not None:
            response_text = getattr(response, "text", None)
            if isinstance(response_text, str):
                message_parts.append(response_text)
            response_content = getattr(response, "content", None)
            if isinstance(response_content, (str, bytes)):
                message_parts.append(
                    response_content.decode("utf-8", errors="ignore")
                    if isinstance(response_content, bytes)
                    else response_content
                )

        message = " ".join(message_parts).lower()
        return any(marker in message for marker in RATE_LIMIT_MARKERS)

    def run_rate_limited_tasks(
        tasks: list[tuple[Any, Any]],
        worker_fn: Callable[[Any], TResult],
        task_label: str,
        max_workers: int,
        max_retries: int,
        base_sleep_seconds: float,
        max_sleep_seconds: float,
    ) -> dict[Any, TResult]:
        """Run tasks with bounded concurrency and adaptive rate-limit retries.

        Args:
            tasks (list[tuple[Any, Any]]): Items as (task_id, payload).
            worker_fn (Callable[[Any], TResult]): Function applied to each payload.
            task_label (str): Label used in logs.
            max_workers (int): Maximum worker count.
            max_retries (int): Maximum retries per task when rate-limited.
            base_sleep_seconds (float): Initial backoff delay.
            max_sleep_seconds (float): Maximum backoff delay.

        Returns:
            dict[Any, TResult]: Mapping from task_id to worker result.
        """
        if not tasks:
            return {}
        workers = max(1, max_workers)
        results: dict[Any, TResult] = {}
        pending: list[tuple[Any, Any, int]] = [
            (task_id, payload, 0) for task_id, payload in tasks
        ]
        round_id = 0

        while pending:
            round_id += 1
            current_batch = pending
            pending = []
            logger.info(
                f"{task_label}: processing {len(current_batch)} task(s) with {workers} worker(s), round {round_id}"
            )
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(worker_fn, payload): (task_id, payload, retry_count)
                    for task_id, payload, retry_count in current_batch
                }
                for future in as_completed(future_map):
                    task_id, payload, retry_count = future_map[future]
                    try:
                        results[task_id] = future.result()
                    except Exception as exc:
                        if is_rate_limit_error(exc) and retry_count < max_retries:
                            pending.append((task_id, payload, retry_count + 1))
                        else:
                            raise

            if pending:
                highest_retry = max(retry_count for _, _, retry_count in pending)
                sleep_seconds = min(
                    max_sleep_seconds,
                    max(base_sleep_seconds, base_sleep_seconds * (2**highest_retry)),
                )
                logger.warning(
                    f"{task_label}: hit rate limits for {len(pending)} task(s); backing off for {sleep_seconds:.1f}s and reducing workers."
                )
                workers = max(1, workers // 2)
                time.sleep(sleep_seconds)
            elif workers < max_workers:
                workers += 1
        return results

    analysis_max_workers = settings.llm_analysis_max_workers
    validation_max_workers = settings.llm_validation_max_workers
    rate_limit_max_retries = settings.llm_rate_limit_max_retries
    rate_limit_base_sleep_seconds = settings.llm_rate_limit_base_sleep_seconds
    rate_limit_max_sleep_seconds = settings.llm_rate_limit_max_sleep_seconds

    logger.info(
        "LLM concurrency settings: "
        f"analysis_workers={analysis_max_workers}, validation_workers={validation_max_workers}, "
        f"max_retries={rate_limit_max_retries}, sleep={rate_limit_base_sleep_seconds:.1f}-{rate_limit_max_sleep_seconds:.1f}s"
    )

    def load_existing_csv(path: Path, columns: list[str]) -> pd.DataFrame:
        """Load CSV if present, otherwise return an empty DataFrame with columns.

        Args:
            path (Path): CSV path.
            columns (list[str]): Columns to use when file does not exist.

        Returns:
            pd.DataFrame: Loaded or empty DataFrame.
        """
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception as e:
                logger.warning(
                    f"Failed to read existing CSV {path}: {e}\n{traceback.format_exc()}. Starting fresh."
                )
        return pd.DataFrame(columns=columns)

    def build_key_set(df: pd.DataFrame, key_columns: list[str]) -> set[tuple[Any, ...]]:
        """Build a set of key tuples from a DataFrame."""
        if df.empty or any(col not in df.columns for col in key_columns):
            return set()
        return {
            tuple(row) for row in df[key_columns].itertuples(index=False, name=None)
        }

    def align_df_columns(
        df: pd.DataFrame, known_columns: list[str] | None
    ) -> pd.DataFrame:
        """Align DataFrame columns to match an existing CSV schema."""
        if known_columns is None:
            return df
        for column in known_columns:
            if column not in df.columns:
                df[column] = pd.NA
        ordered_columns = known_columns + [
            column for column in df.columns if column not in known_columns
        ]
        return df.reindex(columns=ordered_columns)

    def append_unique_df(
        df: pd.DataFrame,
        path: Path,
        key_columns: list[str],
        seen_keys: set[tuple[Any, ...]],
        known_columns: list[str] | None = None,
    ) -> int:
        """Append only unseen rows (based on key columns) to a CSV.

        Args:
            df (pd.DataFrame): Candidate rows to append.
            path (Path): Destination CSV path.
            key_columns (list[str]): Columns that define row uniqueness.
            seen_keys (set[tuple[Any, ...]]): In-memory key registry.
            known_columns (list[str] | None): Optional output column order.

        Returns:
            int: Number of rows appended.
        """
        if df.empty:
            return 0
        if any(col not in df.columns for col in key_columns):
            missing = [col for col in key_columns if col not in df.columns]
            raise KeyError(f"Missing key column(s) {missing} when appending to {path}")

        filtered_indices: list[int] = []
        fresh_keys: list[tuple[Any, ...]] = []
        batch_seen: set[tuple[Any, ...]] = set()
        for idx, key in enumerate(df[key_columns].itertuples(index=False, name=None)):
            if key in seen_keys or key in batch_seen:
                continue
            filtered_indices.append(idx)
            fresh_keys.append(key)
            batch_seen.add(key)

        if not filtered_indices:
            return 0

        filtered_df = df.iloc[filtered_indices].copy()
        filtered_df = align_df_columns(filtered_df, known_columns=known_columns)
        write_header = (not path.exists()) or path.stat().st_size == 0
        filtered_df.to_csv(path, mode="a", index=False, header=write_header)
        seen_keys.update(fresh_keys)
        return len(filtered_df)

    def parse_list_field(value: object) -> list[str]:
        """Parse list-like field from CSV cell value into list[str]."""
        if isinstance(value, list):
            return [str(item) for item in value]
        if value is None or pd.isna(value):
            return []
        if isinstance(value, str):
            text = value.strip()
            if text == "" or text.lower() in {"nan", "none"}:
                return []
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return [text]
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
            if parsed is None:
                return []
            return [str(parsed)]
        return [str(value)]

    def parse_bool_field(value: object) -> bool:
        """Parse bool-like CSV field into bool."""
        if isinstance(value, bool):
            return value
        if value is None or pd.isna(value):
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    def parse_float_field(value: object, default: float = 0.0) -> float:
        """Parse float-like CSV field into float with fallback."""
        if value is None or pd.isna(value):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def parse_sentiment(value: object) -> str:
        """Parse sentiment value constrained to expected labels."""
        if isinstance(value, str) and value in {"positive", "negative", "neutral"}:
            return value
        return "neutral"

    def to_manifesto_chunk_analysis(row: Any) -> ManifestoChunkAnalysis:
        """Convert a CSV row into ManifestoChunkAnalysis."""
        return ManifestoChunkAnalysis(
            policy_proposals=parse_list_field(getattr(row, "policy_proposals", [])),
            sentiment=parse_sentiment(getattr(row, "sentiment", "neutral")),
            topic=(
                getattr(row, "topic", "Other")
                if isinstance(getattr(row, "topic", None), str)
                else "Other"
            ),
            hate_speech=HateSpeechDetection(
                is_hate_speech=parse_bool_field(
                    getattr(row, "hate_speech_is_hate_speech", False)
                ),
                reason=(
                    getattr(row, "hate_speech_reason", "")
                    if isinstance(getattr(row, "hate_speech_reason", None), str)
                    else ""
                ),
                targeted_groups=parse_list_field(
                    getattr(row, "hate_speech_targeted_groups", [])
                ),
            ),
            political_compass=PoliticalCompass(
                economic=parse_float_field(
                    getattr(row, "political_compass_economic", 0.0)
                ),
                social=parse_float_field(getattr(row, "political_compass_social", 0.0)),
            ),
        )

    def get_processed_files(
        df: pd.DataFrame, column_name: str = "pdf_file"
    ) -> set[str]:
        """Return unique non-null file names from a DataFrame column."""
        if column_name not in df.columns:
            return set()
        return set(df[column_name].dropna().astype(str).tolist())

    def load_text_items_for_file(
        existing_df: pd.DataFrame,
        filename: str,
        index_column: str,
        value_column: str,
    ) -> list[str]:
        """Load sorted text items for one PDF from an existing table.

        Args:
            existing_df (pd.DataFrame): Existing table DataFrame.
            filename (str): PDF filename.
            index_column (str): Index/order column name.
            value_column (str): Text content column name.

        Returns:
            list[str]: Sorted text values for the file.
        """
        tmp = existing_df[existing_df["pdf_file"] == filename].sort_values(index_column)
        return [
            value if isinstance(value, str) else ""
            for value in tmp[value_column].tolist()
        ]

    def save_indexed_text_items(
        filename: str,
        items: list[str],
        index_column: str,
        value_column: str,
        csv_path: Path,
        seen_keys: set[tuple[Any, ...]],
    ) -> int:
        """Persist indexed text rows for a PDF file to CSV.

        Args:
            filename (str): PDF filename.
            items (list[str]): Text items in order.
            index_column (str): Index/order column name.
            value_column (str): Text content column name.
            csv_path (Path): Destination CSV path.
            seen_keys (set[tuple[Any, ...]]): In-memory key registry.

        Returns:
            int: Number of appended rows.
        """
        rows = [
            {"pdf_file": filename, index_column: index, value_column: value}
            for index, value in enumerate(items)
        ]
        return append_unique_df(
            pd.DataFrame(rows),
            csv_path,
            key_columns=["pdf_file", index_column],
            seen_keys=seen_keys,
        )

    # Prepare CSV file paths for iterative saving
    csv_party = output_folder / "party_names.csv"
    csv_analysis = output_folder / "chunk_analysis.csv"
    csv_validation = output_folder / "scientific_validations.csv"
    csv_mapping = output_folder / "topic_mapping.csv"
    csv_unified = output_folder / "unified_topics.csv"
    csv_parsed_pages = output_folder / "parsed_pages.csv"
    csv_chunks = output_folder / "chunks.csv"

    pages_existing = load_existing_csv(
        csv_parsed_pages, ["pdf_file", "page_index", "page_content"]
    )
    chunks_existing = load_existing_csv(
        csv_chunks, ["pdf_file", "chunk_index", "chunk_content"]
    )
    party_existing = load_existing_csv(csv_party, ["pdf_file"])
    analysis_existing = load_existing_csv(
        csv_analysis,
        ANALYSIS_TABLE_COLUMNS,
    )
    validation_existing = load_existing_csv(
        csv_validation, ["pdf_file", "chunk_index", "proposal_index"]
    )
    mapping_existing = load_existing_csv(
        csv_mapping, ["original_topic", "unified_topic"]
    )
    unified_existing = load_existing_csv(csv_unified, ["unified_topic"])

    pages_seen_keys = build_key_set(pages_existing, ["pdf_file", "page_index"])
    chunks_seen_keys = build_key_set(chunks_existing, ["pdf_file", "chunk_index"])
    analysis_seen_keys = build_key_set(analysis_existing, ["pdf_file", "chunk_index"])
    validation_seen_keys = build_key_set(
        validation_existing, ["pdf_file", "chunk_index", "proposal_index"]
    )
    mapping_seen_keys = build_key_set(mapping_existing, ["original_topic"])
    unified_seen_keys = build_key_set(unified_existing, ["unified_topic"])

    pages_done_files = get_processed_files(pages_existing)
    chunks_done_files = get_processed_files(chunks_existing)
    party_seen_keys = build_key_set(party_existing, ["pdf_file"])

    analysis_columns = analysis_existing.columns.tolist()
    validation_columns = validation_existing.columns.tolist()

    pdf_processor = MistralPDFProcessor()
    openai_pdf_processor = OpenAIPDFProcessor()
    text_chunker = OpenAITextChunker()
    party_extractor = OpenAIPartyNameExtractor()
    analyzer = OpenAIStructuredChunkAnalyzer()
    validator = GeminiScientificValidator()
    marker_processor = MarkerPDFProcessor()
    markdown_chunker = MarkdownTextChunker()

    # Precompute PDF file list once
    pdf_filenames = sorted(
        [
            p.name
            for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() == ".pdf"
        ]
    )
    if not pdf_filenames:
        logger.warning(f"No PDF files found in input_folder={input_path}")
        return

    # Batch-parse PDFs that still need page extraction
    mistral_batch_pages: dict[str, list[str]] = {}
    missing_pages_files = [
        filename for filename in pdf_filenames if filename not in pages_done_files
    ]
    if missing_pages_files:
        try:
            logger.info(
                f"Step: Mistral batch PDF parsing for {len(missing_pages_files)} file(s)"
            )
            missing_paths = [input_path / filename for filename in missing_pages_files]
            batch_results = pdf_processor.process_batch(missing_paths)
            mistral_batch_pages = {
                filename: pages
                for filename, pages in zip(
                    missing_pages_files, batch_results, strict=False
                )
            }
            logger.info(
                f"Mistral batch parsing completed for {len(mistral_batch_pages)} file(s)"
            )
        except Exception as e:
            logger.warning(
                f"Mistral batch PDF parsing failed: {e}\n{traceback.format_exc()}. Falling back to per-file parsers."
            )

    # Iterate over PDF files with a progress bar
    for filename in tqdm(pdf_filenames, desc="PDFs to process", unit="file"):
        pdf_path = input_path / filename
        logger.info(f"Processing PDF: {pdf_path}")

        try:
            # Step 1: Parse PDF into markdown pages
            if filename in pages_done_files:
                logger.info(f"Parsed pages for {filename} exist, loading from CSV")
                pages = load_text_items_for_file(
                    existing_df=pages_existing,
                    filename=filename,
                    index_column="page_index",
                    value_column="page_content",
                )
            elif filename in mistral_batch_pages:
                logger.info(f"Using Mistral batch parsed pages for {filename}")
                pages = mistral_batch_pages[filename]
                added = save_indexed_text_items(
                    filename=filename,
                    items=pages,
                    index_column="page_index",
                    value_column="page_content",
                    csv_path=csv_parsed_pages,
                    seen_keys=pages_seen_keys,
                )
                if added:
                    logger.info(
                        f"Saved {added} parsed page row(s) to {csv_parsed_pages}"
                    )
                pages_done_files.add(filename)
            else:
                logger.info(f"Step: PDF parsing for {filename}")
                try:
                    pages = pdf_processor.process(pdf_path)
                except Exception as e:
                    logger.warning(
                        f"MistralPDFProcessor failed for {filename}: {e}\n{traceback.format_exc()}. Falling back to OpenAIPDFProcessor."
                    )
                    try:
                        pages = openai_pdf_processor.process(pdf_path)
                    except Exception as mistral_error:
                        logger.warning(
                            f"MistralPDFProcessor failed for {filename}: {mistral_error}\n{traceback.format_exc()}. Falling back to MarkerPDFProcessor."
                        )
                        pages = marker_processor.process(pdf_path)
                logger.info(f"Parsed {len(pages)} pages from {filename}")
                added = save_indexed_text_items(
                    filename=filename,
                    items=pages,
                    index_column="page_index",
                    value_column="page_content",
                    csv_path=csv_parsed_pages,
                    seen_keys=pages_seen_keys,
                )
                if added:
                    logger.info(
                        f"Saved {added} parsed page row(s) to {csv_parsed_pages}"
                    )
                pages_done_files.add(filename)

            # Step 2: Chunk the text into semantic chunks
            if filename in chunks_done_files:
                logger.info(f"Chunks for {filename} exist, loading from CSV")
                chunks = load_text_items_for_file(
                    existing_df=chunks_existing,
                    filename=filename,
                    index_column="chunk_index",
                    value_column="chunk_content",
                )
            else:
                logger.info(f"Step: Text chunking for {filename}")
                try:
                    chunks = text_chunker.process(pages)
                except Exception as e:
                    logger.warning(
                        f"OpenAITextChunker failed for {filename}: {e}\n{traceback.format_exc()}. Falling back to MarkdownTextChunker."
                    )
                    try:
                        chunks = markdown_chunker.process(pages, raw_chunks_only=False)
                    except Exception as e:
                        logger.warning(
                            f"MarkdownTextChunker failed for {filename}: {e}\n{traceback.format_exc()}. Falling back to MarkdownTextChunker without merging similar chunks."
                        )
                        chunks = markdown_chunker.process(pages, raw_chunks_only=True)
                logger.info(f"Generated {len(chunks)} semantic chunks for {filename}")
                added = save_indexed_text_items(
                    filename=filename,
                    items=chunks,
                    index_column="chunk_index",
                    value_column="chunk_content",
                    csv_path=csv_chunks,
                    seen_keys=chunks_seen_keys,
                )
                if added:
                    logger.info(f"Saved {added} chunk row(s) to {csv_chunks}")
                chunks_done_files.add(filename)

            # Step 3: Extract party name
            if (filename,) in party_seen_keys:
                logger.info(f"Party name for {filename} exists, skipping extraction")
            else:
                logger.info(f"Step: Party name extraction for {filename}")
                party_name = party_extractor.extract_party_names(chunks)
                logger.info(
                    f"Extracted party name: {party_name.full_name} (short version: {party_name.short_name})"
                )
                df_party = pd.DataFrame(
                    [
                        {
                            "pdf_file": filename,
                            "full_name": party_name.full_name,
                            "short_name": party_name.short_name,
                            "is_confident": party_name.is_confident,
                        }
                    ]
                )
                added = append_unique_df(
                    df_party,
                    csv_party,
                    key_columns=["pdf_file"],
                    seen_keys=party_seen_keys,
                )
                if added:
                    logger.info(f"Saved party name for {filename} to {csv_party}")

            # Step 4: Analyze each chunk for structured analysis
            chunk_analysis_by_index: dict[int, ManifestoChunkAnalysis] = {}
            df_existing_analysis_for_file = analysis_existing[
                analysis_existing["pdf_file"] == filename
            ]
            if not df_existing_analysis_for_file.empty:
                logger.info(f"Analysis for {filename} exists, loading from CSV")
                df_analysis = df_existing_analysis_for_file.sort_values("chunk_index")
                for row in df_analysis.itertuples():
                    try:
                        chunk_index = int(float(getattr(row, "chunk_index")))
                    except (TypeError, ValueError):
                        logger.warning(
                            f"Skipping malformed chunk_index in analysis CSV for {filename}: {row}"
                        )
                        continue
                    chunk_analysis_by_index[chunk_index] = to_manifesto_chunk_analysis(
                        row
                    )

            missing_chunk_indices = [
                idx
                for idx in range(len(chunks))
                if (filename, idx) not in analysis_seen_keys
            ]
            if missing_chunk_indices:
                logger.info(
                    f"Step: Structured analysis of {len(missing_chunk_indices)} missing chunk(s) for {filename}"
                )
                analysis_tasks: list[tuple[int, str]] = [
                    (idx, chunks[idx]) for idx in missing_chunk_indices
                ]
                analysis_results = run_rate_limited_tasks(
                    tasks=analysis_tasks,
                    worker_fn=analyzer.process,
                    task_label=f"Structured analysis ({filename})",
                    max_workers=analysis_max_workers,
                    max_retries=rate_limit_max_retries,
                    base_sleep_seconds=rate_limit_base_sleep_seconds,
                    max_sleep_seconds=rate_limit_max_sleep_seconds,
                )
                chunk_indices = sorted(analysis_results.keys())
                chunk_analysis_models = [analysis_results[idx] for idx in chunk_indices]
                for idx, analysis in analysis_results.items():
                    chunk_analysis_by_index[idx] = analysis
                logger.info(
                    f"Completed structured analysis of {len(missing_chunk_indices)} missing chunk(s) for {filename}"
                )
                if chunk_analysis_models:
                    df_analysis = convert_pydantic_to_dataframe(chunk_analysis_models)
                    # Add metadata columns
                    df_analysis.insert(0, "pdf_file", filename)
                    df_analysis.insert(1, "chunk_index", chunk_indices)
                    df_analysis = align_df_columns(
                        df_analysis, known_columns=analysis_columns
                    )
                    added = append_unique_df(
                        df_analysis,
                        csv_analysis,
                        key_columns=["pdf_file", "chunk_index"],
                        seen_keys=analysis_seen_keys,
                        known_columns=analysis_columns,
                    )
                    if added:
                        logger.info(f"Saved {added} analysis row(s) to {csv_analysis}")
            else:
                logger.info(f"No missing analysis rows for {filename}")

            chunk_analysis_entries: list[tuple[int, ManifestoChunkAnalysis]] = sorted(
                chunk_analysis_by_index.items()
            )
            if not chunk_analysis_entries:
                logger.warning(
                    f"No chunk analyses available for {filename}; skipping scientific validation."
                )

            # Step 5: Validate each policy proposal scientifically
            logger.info(
                f"Step: Scientific validation of policy proposals for {filename}"
            )
            validation_rows: list[dict[str, object]] = []
            proposal_tasks: list[tuple[tuple[int, int], tuple[int, int, str]]] = []
            for chunk_index, model in chunk_analysis_entries:
                for proposal_index, proposal in enumerate(model.policy_proposals):
                    validation_key = (filename, chunk_index, proposal_index)
                    if validation_key in validation_seen_keys:
                        continue
                    proposal_tasks.append(
                        (
                            (chunk_index, proposal_index),
                            (chunk_index, proposal_index, proposal),
                        )
                    )

            if proposal_tasks:

                def validate_proposal(
                    payload: tuple[int, int, str],
                ) -> tuple[Any, list[Any]]:
                    _chunk_index, _proposal_index, proposal = payload
                    return validator.process(proposal)

                validation_results = run_rate_limited_tasks(
                    tasks=proposal_tasks,
                    worker_fn=validate_proposal,
                    task_label=f"Scientific validation ({filename})",
                    max_workers=validation_max_workers,
                    max_retries=rate_limit_max_retries,
                    base_sleep_seconds=rate_limit_base_sleep_seconds,
                    max_sleep_seconds=rate_limit_max_sleep_seconds,
                )

                for (chunk_index, proposal_index), (validation, citations) in sorted(
                    validation_results.items()
                ):
                    proposal = chunk_analysis_by_index[chunk_index].policy_proposals[
                        proposal_index
                    ]
                    val_data = validation.model_dump()
                    validation_rows.append(
                        {
                            "pdf_file": filename,
                            "chunk_index": chunk_index,
                            "proposal_index": proposal_index,
                            "proposal": proposal,
                            **val_data,
                            "citations": citations,
                        }
                    )
            if validation_rows:
                validation_df = expand_dict_columns(pd.DataFrame(validation_rows))
                validation_df = align_df_columns(
                    validation_df, known_columns=validation_columns
                )
                added = append_unique_df(
                    validation_df,
                    csv_validation,
                    key_columns=["pdf_file", "chunk_index", "proposal_index"],
                    seen_keys=validation_seen_keys,
                    known_columns=validation_columns,
                )
                if added:
                    logger.info(f"Saved {added} validation row(s) to {csv_validation}")
            else:
                logger.info(f"No new validations needed for {filename}")

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}\n{traceback.format_exc()}")

    # Step 6: Topic unification across all PDFs
    if mapping_existing.empty or unified_existing.empty:
        logger.info("Step: Topic unification across all processed PDFs")
        combined_analysis_df = load_existing_csv(
            csv_analysis, ["pdf_file", "chunk_index", "topic"]
        )
        if combined_analysis_df.empty or "topic" not in combined_analysis_df.columns:
            logger.warning("No chunk analysis data found. Skipping topic unification.")
        else:
            combined_analysis_df["topic"] = (
                combined_analysis_df["topic"].fillna("Other").astype(str)
            )
            topic_counts: dict[str, int] = (
                combined_analysis_df["topic"].value_counts().to_dict()
            )
            if not topic_counts:
                logger.warning(
                    "No topics available for unification. Skipping topic unification."
                )
            else:
                # Run unifier on aggregated topics
                topic_unifier = OpenAITopicUnifier()
                unified_output = topic_unifier.process(topic_counts)

                # Build mapping records and dictionary for vectorized mapping
                mapping_records = []
                mapping_dict: dict[str, str] = {}
                for unified in unified_output.unified_topics:
                    for original in unified.original_topics:
                        mapping_records.append(
                            {"original_topic": original, "unified_topic": unified.name}
                        )
                        mapping_dict[original] = unified.name

                combined_analysis_df["unified_topic"] = (
                    combined_analysis_df["topic"].map(mapping_dict).fillna("Other")
                )
                combined_analysis_df.to_csv(csv_analysis, index=False)
                logger.success("Saved chunk_analysis.csv with unified_topic column")

                unified_topics_records = [
                    {"unified_topic": ut.name} for ut in unified_output.unified_topics
                ]
                added_unified = append_unique_df(
                    pd.DataFrame(unified_topics_records),
                    csv_unified,
                    key_columns=["unified_topic"],
                    seen_keys=unified_seen_keys,
                )
                if added_unified:
                    logger.success(
                        f"Saved {added_unified} row(s) to unified_topics.csv"
                    )

                added_mapping = append_unique_df(
                    pd.DataFrame(mapping_records),
                    csv_mapping,
                    key_columns=["original_topic"],
                    seen_keys=mapping_seen_keys,
                )
                if added_mapping:
                    logger.success(f"Saved {added_mapping} row(s) to topic_mapping.csv")
    else:
        logger.info(
            "Applying existing topic mapping to chunk_analysis.csv using loaded mapping"
        )
        # Load existing analysis table and apply mapping
        analysis_df = load_existing_csv(
            csv_analysis, ["pdf_file", "chunk_index", "topic"]
        )
        if analysis_df.empty or "topic" not in analysis_df.columns:
            logger.warning(
                "No chunk_analysis.csv data found; skipping topic mapping application."
            )
        else:
            analysis_df["topic"] = analysis_df["topic"].fillna("Other").astype(str)
            # Build mapping dictionary from loaded CSV
            mapping_dict = mapping_existing.set_index("original_topic")[
                "unified_topic"
            ].to_dict()
            analysis_df["unified_topic"] = (
                analysis_df["topic"].map(mapping_dict).fillna("Other")
            )
            # Save updated chunk analysis with unified topics, overwriting existing file
            analysis_df.to_csv(csv_analysis, index=False)
            logger.success(
                "Updated chunk_analysis.csv with unified_topic column from existing mappings"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the PDF processing pipeline.")
    parser.add_argument(
        "input_folder", type=str, help="Path to the folder containing PDF files."
    )
    args = parser.parse_args()

    process_pdfs(args.input_folder)
