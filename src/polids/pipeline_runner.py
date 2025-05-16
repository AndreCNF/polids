import os
import pandas as pd  # type: ignore[import]
from loguru import logger
from pathlib import Path
from tqdm.auto import tqdm  # type: ignore[import]
import traceback

from polids.pdf_processing.openai import OpenAIPDFProcessor
from polids.pdf_processing.marker import MarkerPDFProcessor
from polids.structured_analysis.base import (
    HateSpeechDetection,
    ManifestoChunkAnalysis,
    PoliticalCompass,
)
from polids.text_chunking.openai import OpenAITextChunker
from polids.text_chunking.markdown_chunker import MarkdownTextChunker
from polids.party_name_extraction.openai import OpenAIPartyNameExtractor
from polids.structured_analysis.openai import OpenAIStructuredChunkAnalyzer
from polids.scientific_validation.perplexity import PerplexityScientificValidator
from polids.scientific_validation.openai import OpenAIScientificValidator
from polids.topic_unification.openai import OpenAITopicUnifier
from polids.utils.pandas import convert_pydantic_to_dataframe, expand_dict_columns


def process_pdfs(input_folder: str) -> None:
    """
    Process PDFs in the input folder through the pipeline and save results as CSV tables.

    Args:
        input_folder (str): Path to the folder containing PDF files.
    """
    output_folder = os.path.join(input_folder, "output")
    os.makedirs(output_folder, exist_ok=True)

    # Helper function: save DataFrame uniquely to CSV, avoiding duplicate rows
    def save_unique(df: pd.DataFrame, path: Path) -> None:
        """
        Save DataFrame to CSV at path, reading existing CSV if present and
        appending only new unique rows.
        """
        if path.exists():
            try:
                existing_df = pd.read_csv(path)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                # Convert list entries to tuples to avoid unhashable type errors
                for column in combined_df.columns:
                    combined_df[column] = combined_df[column].apply(
                        lambda x: tuple(x) if isinstance(x, list) else x
                    )
                combined_df.drop_duplicates(inplace=True)
                combined_df.to_csv(path, index=False)
            except Exception as e:
                logger.warning(
                    f"Failed to read existing CSV {path}: {e}\n{traceback.format_exc()}. Overwriting with new data."
                )
                df.to_csv(path, index=False)
        else:
            df.to_csv(path, index=False)

    # Prepare CSV file paths for iterative saving
    csv_party = Path(output_folder) / "party_names.csv"
    csv_analysis = Path(output_folder) / "chunk_analysis.csv"
    csv_validation = Path(output_folder) / "scientific_validations.csv"
    csv_mapping = Path(output_folder) / "topic_mapping.csv"
    csv_unified = Path(output_folder) / "unified_topics.csv"
    csv_parsed_pages = Path(output_folder) / "parsed_pages.csv"
    csv_chunks = Path(output_folder) / "chunks.csv"

    # Load existing data to skip completed steps
    def load_existing_csv(path: Path, columns: list[str]) -> pd.DataFrame:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception as e:
                logger.warning(
                    f"Failed to read existing CSV {path}: {e}\n{traceback.format_exc()}. Starting fresh."
                )
        return pd.DataFrame(columns=columns)

    pages_existing = load_existing_csv(
        csv_parsed_pages, ["pdf_file", "page_index", "page_content"]
    )
    chunks_existing = load_existing_csv(
        csv_chunks, ["pdf_file", "chunk_index", "chunk_content"]
    )
    party_existing = load_existing_csv(csv_party, ["pdf_file"])
    analysis_existing = load_existing_csv(csv_analysis, ["pdf_file", "chunk_index"])
    validation_existing = load_existing_csv(
        csv_validation, ["pdf_file", "chunk_index", "proposal_index"]
    )
    mapping_existing = load_existing_csv(
        csv_mapping, ["original_topic", "unified_topic"]
    )
    unified_existing = load_existing_csv(csv_unified, ["unified_topic"])

    # Prepare accumulators for multiple tables
    party_records: list[dict[str, object]] = []
    analysis_dfs: list[pd.DataFrame] = []
    validation_records: list[dict[str, object]] = []
    mapping_records: list[dict[str, str]] = []
    unified_topics_records: list[dict[str, str]] = []
    pages_records: list[dict[str, object]] = []
    chunks_records: list[dict[str, object]] = []

    # Iterate over PDF files with a progress bar
    for filename in tqdm(os.listdir(input_folder), desc="PDFs to process", unit="file"):
        if not filename.endswith(".pdf"):
            logger.warning(f"Skipping non-PDF file: {filename}")
            continue

        pdf_path = Path(input_folder) / filename
        logger.info(f"Processing PDF: {pdf_path}")

        try:
            # Initialize pipeline components
            pdf_processor = OpenAIPDFProcessor()
            text_chunker = OpenAITextChunker()
            party_extractor = OpenAIPartyNameExtractor()
            analyzer = OpenAIStructuredChunkAnalyzer()
            validator = PerplexityScientificValidator()

            # Step 1: Parse PDF into markdown pages
            if filename in pages_existing["pdf_file"].values:
                logger.info(f"Parsed pages for {filename} exist, loading from CSV")
                tmp = pages_existing[
                    pages_existing["pdf_file"] == filename
                ].sort_values("page_index")
                pages = tmp["page_content"].tolist()
            else:
                logger.info(f"Step: PDF parsing for {filename}")
                try:
                    pages = pdf_processor.process(pdf_path)
                except Exception as e:
                    logger.warning(
                        f"OpenAIPDFProcessor failed for {filename}: {e}\n{traceback.format_exc()}. Falling back to MarkerPDFProcessor."
                    )
                    marker_processor = MarkerPDFProcessor()
                    pages = marker_processor.process(pdf_path)
                logger.info(f"Parsed {len(pages)} pages from {filename}")
                # Save parsed pages to CSV
                for page_index, page_content in enumerate(pages):
                    pages_records.append(
                        {
                            "pdf_file": filename,
                            "page_index": page_index,
                            "page_content": page_content,
                        }
                    )
                save_unique(pd.DataFrame(pages_records), csv_parsed_pages)
                logger.info(f"Saved parsed pages to {csv_parsed_pages}")

            # Step 2: Chunk the text into semantic chunks
            if filename in chunks_existing["pdf_file"].values:
                logger.info(f"Chunks for {filename} exist, loading from CSV")
                tmp = chunks_existing[
                    chunks_existing["pdf_file"] == filename
                ].sort_values("chunk_index")
                chunks = tmp["chunk_content"].tolist()
            else:
                logger.info(f"Step: Text chunking for {filename}")
                try:
                    chunks = text_chunker.process(pages)
                except Exception as e:
                    logger.warning(
                        f"OpenAITextChunker failed for {filename}: {e}\n{traceback.format_exc()}. Falling back to MarkdownTextChunker."
                    )
                    try:
                        markdown_chunker = MarkdownTextChunker()
                        chunks = markdown_chunker.process(pages, raw_chunks_only=False)
                    except Exception as e:
                        logger.warning(
                            f"MarkdownTextChunker failed for {filename}: {e}\n{traceback.format_exc()}. Falling back to MarkdownTextChunker without merging similar chunks."
                        )
                        markdown_chunker = MarkdownTextChunker()
                        chunks = markdown_chunker.process(pages, raw_chunks_only=True)
                logger.info(f"Generated {len(chunks)} semantic chunks for {filename}")
                # Save chunks to CSV
                for chunk_index, chunk_content in enumerate(chunks):
                    chunks_records.append(
                        {
                            "pdf_file": filename,
                            "chunk_index": chunk_index,
                            "chunk_content": chunk_content,
                        }
                    )
                save_unique(pd.DataFrame(chunks_records), csv_chunks)
                logger.info(f"Saved chunks to {csv_chunks}")

            # Step 3: Extract party name
            if filename in party_existing["pdf_file"].values:
                logger.info(f"Party name for {filename} exists, skipping extraction")
            else:
                logger.info(f"Step: Party name extraction for {filename}")
                party_name = party_extractor.extract_party_names(chunks)
                logger.info(
                    f"Extracted party name: {party_name.full_name} (short version: {party_name.short_name})"
                )
                party_records.append(
                    {
                        "pdf_file": filename,
                        "full_name": party_name.full_name,
                        "short_name": party_name.short_name,
                        "is_confident": party_name.is_confident,
                    }
                )
                # Iteratively save party names
                save_unique(pd.DataFrame(party_records), csv_party)
                logger.info(
                    f"Iteratively saved {len(party_records)} party name records to party_names.csv"
                )

            # Step 4: Analyze each chunk for structured analysis
            if filename in analysis_existing["pdf_file"].values:
                logger.info(f"Analysis for {filename} exists, loading from CSV")
                df_analysis = analysis_existing[
                    analysis_existing["pdf_file"] == filename
                ]
                analysis_dfs.append(df_analysis)
                # Convert DataFrame rows to ManifestoChunkAnalysis models,
                # which will be used for scientific validation
                chunk_analysis_models = [
                    ManifestoChunkAnalysis(
                        policy_proposals=row.policy_proposals
                        if isinstance(row.policy_proposals, list)
                        else eval(row.policy_proposals),
                        sentiment=row.sentiment,
                        topic=row.topic,
                        hate_speech=HateSpeechDetection(
                            is_hate_speech=row.hate_speech_is_hate_speech,
                            reason=row.hate_speech_reason
                            if isinstance(row.hate_speech_reason, str)
                            else "",
                            targeted_groups=row.hate_speech_targeted_groups
                            if isinstance(row.hate_speech_targeted_groups, list)
                            else eval(row.hate_speech_targeted_groups),
                        ),
                        political_compass=PoliticalCompass(
                            economic=row.political_compass_economic,
                            social=row.political_compass_social,
                        ),
                    )
                    for row in df_analysis.itertuples()
                ]
            else:
                logger.info(f"Step: Structured analysis of chunks for {filename}")
                chunk_analysis_models = []
                for idx, chunk in tqdm(
                    enumerate(chunks),
                    total=len(chunks),
                    desc="Analyzing chunks",
                    unit="chunk",
                    leave=False,
                ):
                    analysis = analyzer.process(chunk)
                    chunk_analysis_models.append(analysis)
                logger.info(
                    f"Completed structured analysis of {len(chunks)} chunks for {filename}"
                )
                df_analysis = convert_pydantic_to_dataframe(chunk_analysis_models)  # type: ignore
                # Add metadata columns
                df_analysis.insert(0, "pdf_file", filename)
                df_analysis.insert(1, "chunk_index", range(len(chunks)))
                analysis_dfs.append(df_analysis)
                # Iteratively save chunk analysis
                save_unique(pd.concat(analysis_dfs, ignore_index=True), csv_analysis)
                logger.info(
                    f"Iteratively saved structured analysis for {len(analysis_dfs)} PDFs to chunk_analysis.csv"
                )

            # Step 5: Validate each policy proposal scientifically
            if filename in validation_existing["pdf_file"].values:
                logger.info(
                    f"Validation for {filename} exists, skipping scientific validation"
                )
            else:
                logger.info(
                    f"Step: Scientific validation of policy proposals for {filename}"
                )
                for idx, model in tqdm(
                    enumerate(chunk_analysis_models),
                    total=len(chunk_analysis_models),
                    desc="Validating proposals",
                    unit="chunk",
                    leave=False,
                ):
                    if model.policy_proposals:
                        for p_idx, proposal in enumerate(
                            tqdm(
                                model.policy_proposals,
                                desc=f"Validating proposals in chunk {idx}",
                                unit="proposal",
                                leave=False,
                            )
                        ):
                            try:
                                validation, citations = validator.process(proposal)
                            except Exception as e:
                                logger.warning(
                                    f"PerplexityScientificValidator failed for proposal {p_idx} in chunk {idx} of {filename}: {e}\n{traceback.format_exc()}. Falling back to OpenAIScientificValidator."
                                )
                                openai_validator = OpenAIScientificValidator()
                                validation, citations = openai_validator.process(
                                    proposal
                                )
                            val_data = validation.model_dump()
                            validation_records.append(
                                {
                                    "pdf_file": filename,
                                    "chunk_index": idx,
                                    "proposal_index": p_idx,
                                    "proposal": proposal,
                                    **val_data,
                                    "citations": citations,
                                }
                            )
                # Iteratively save scientific validations
                if validation_records:
                    save_unique(
                        expand_dict_columns(pd.DataFrame(validation_records)),
                        csv_validation,
                    )
                    logger.info(
                        f"Iteratively saved {len(validation_records)} scientific validation records to scientific_validations.csv"
                    )

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}\n{traceback.format_exc()}")

    # Step 6: Topic unification across all PDFs
    if mapping_existing.empty or unified_existing.empty:
        logger.info("Step: Topic unification across all processed PDFs")
        # Aggregate topic frequencies from all chunk analyses
        combined_analysis_df = pd.concat(analysis_dfs, ignore_index=True)
        topic_counts: dict[str, int] = (
            combined_analysis_df["topic"].value_counts().to_dict()
        )
        # Run unifier on aggregated topics
        topic_unifier = OpenAITopicUnifier()
        unified_output = topic_unifier.process(topic_counts)
        # Build mapping records
        mapping_records = []
        for unified in unified_output.unified_topics:
            for original in unified.original_topics:
                mapping_records.append(
                    {"original_topic": original, "unified_topic": unified.name}
                )
        # Ensure topic column has no NaN or non-string values
        combined_analysis_df["topic"] = combined_analysis_df["topic"].fillna("Other")
        # Map each input topic to its unified topic
        tqdm.pandas(desc="Mapping topics to unified topics")
        combined_analysis_df["unified_topic"] = combined_analysis_df[
            "topic"
        ].progress_apply(topic_unifier.map_input_topic_to_unified_topic)
        # Save updated chunk analysis with unified topics
        save_unique(combined_analysis_df, csv_analysis)
        logger.success("Saved chunk_analysis.csv with unified_topic column")

        # Save unified topics list
        unified_topics_records = [
            {"unified_topic": ut.name} for ut in unified_output.unified_topics
        ]
        save_unique(pd.DataFrame(unified_topics_records), csv_unified)
        logger.success("Saved unified_topics.csv")

        # Save topic mapping table
        save_unique(pd.DataFrame(mapping_records), csv_mapping)
        logger.success("Saved topic_mapping.csv")
    else:
        logger.info(
            "Applying existing topic mapping to chunk_analysis.csv using loaded mapping"
        )
        # Load existing analysis table and apply mapping
        analysis_df = pd.read_csv(csv_analysis)
        analysis_df["topic"] = analysis_df["topic"].fillna("Other")
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

    # Save party and validation tables as before
    logger.info("Saving remaining outputs...")
    if party_records:
        save_unique(pd.DataFrame(party_records), csv_party)
        logger.success("Saved party_names.csv")
    if validation_records:
        save_unique(pd.DataFrame(validation_records), csv_validation)
        logger.success("Saved scientific_validations.csv")
    # Save parsed pages and chunks tables
    if pages_records:
        save_unique(pd.DataFrame(pages_records), csv_parsed_pages)
        logger.success("Saved parsed_pages.csv")
    if chunks_records:
        save_unique(pd.DataFrame(chunks_records), csv_chunks)
        logger.success("Saved chunks.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the PDF processing pipeline.")
    parser.add_argument(
        "input_folder", type=str, help="Path to the folder containing PDF files."
    )
    args = parser.parse_args()

    process_pdfs(args.input_folder)
