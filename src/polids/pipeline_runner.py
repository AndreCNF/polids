import os
import pandas as pd  # type: ignore[import]
from loguru import logger
from pathlib import Path
from tqdm.auto import tqdm  # type: ignore[import]

from polids.pdf_processing.openai import OpenAIPDFProcessor
from polids.text_chunking.openai import OpenAITextChunker
from polids.party_name_extraction.openai import OpenAIPartyNameExtractor
from polids.structured_analysis.openai import OpenAIStructuredChunkAnalyzer
from polids.scientific_validation.perplexity import PerplexityScientificValidator
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

    # Prepare CSV file paths for iterative saving
    csv_party = Path(output_folder) / "party_names.csv"
    csv_analysis = Path(output_folder) / "chunk_analysis.csv"
    csv_validation = Path(output_folder) / "scientific_validations.csv"
    csv_mapping = Path(output_folder) / "topic_mapping.csv"
    csv_unified = Path(output_folder) / "unified_topics.csv"

    # Prepare accumulators for multiple tables
    party_records: list[dict[str, object]] = []
    analysis_dfs: list[pd.DataFrame] = []
    validation_records: list[dict[str, object]] = []
    mapping_records: list[dict[str, str]] = []
    unified_topics_records: list[dict[str, str]] = []

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
            logger.info(f"Step: PDF parsing for {filename}")
            pages: list[str] = pdf_processor.process(pdf_path)
            logger.info(f"Parsed {len(pages)} pages from {filename}")

            # Step 2: Chunk the text into semantic chunks
            logger.info(f"Step: Text chunking for {filename}")
            chunks: list[str] = text_chunker.process(pages)
            logger.info(f"Generated {len(chunks)} semantic chunks for {filename}")

            # Step 3: Extract party name
            logger.info(f"Step: Party name extraction for {filename}")
            party_name = party_extractor.extract_party_names(chunks)
            logger.info(
                f"Extracted party name: {party_name.full_name} (confident={party_name.is_confident})"
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
            pd.DataFrame(party_records).to_csv(csv_party, index=False)
            logger.info(
                f"Iteratively saved {len(party_records)} party name records to party_names.csv"
            )

            # Step 4: Analyze each chunk for structured analysis
            logger.info(f"Step: Structured analysis of chunks for {filename}")
            chunk_analysis_models = []
            for idx, chunk in enumerate(
                tqdm(chunks, desc="Analyzing chunks", unit="chunk", leave=False)
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
            pd.concat(analysis_dfs, ignore_index=True).to_csv(csv_analysis, index=False)
            logger.info(
                f"Iteratively saved structured analysis for {len(analysis_dfs)} PDFs to chunk_analysis.csv"
            )

            # Step 5: Validate each policy proposal scientifically
            logger.info(
                f"Step: Scientific validation of policy proposals for {filename}"
            )
            for idx, model in enumerate(chunk_analysis_models):
                if model.policy_proposals:
                    for p_idx, proposal in enumerate(
                        tqdm(
                            model.policy_proposals,
                            desc=f"Validating proposals in chunk {idx}",
                            unit="proposal",
                            leave=False,
                        )
                    ):
                        validation, citations = validator.process(proposal)
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
                expand_dict_columns(pd.DataFrame(validation_records)).to_csv(
                    csv_validation, index=False
                )
                logger.info(
                    f"Iteratively saved {len(validation_records)} scientific validation records to scientific_validations.csv"
                )

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")

    # Step 6: Topic unification across all PDFs
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
    # Add unified_topic column to chunk analysis DataFrame
    combined_analysis_df["unified_topic"] = combined_analysis_df["topic"].apply(
        topic_unifier.map_input_topic_to_unified_topic
    )
    # Save updated chunk analysis with unified topics
    combined_analysis_df.to_csv(csv_analysis, index=False)
    logger.success("Saved chunk_analysis.csv with unified_topic column")

    # Save unified topics list
    unified_topics_records = [
        {"unified_topic": ut.name} for ut in unified_output.unified_topics
    ]
    pd.DataFrame(unified_topics_records).to_csv(csv_unified, index=False)
    logger.success("Saved unified_topics.csv")

    # Save topic mapping table
    pd.DataFrame(mapping_records).to_csv(csv_mapping, index=False)
    logger.success("Saved topic_mapping.csv")

    # Save party and validation tables as before
    logger.info("Saving remaining outputs...")
    if party_records:
        pd.DataFrame(party_records).to_csv(csv_party, index=False)
        logger.success("Saved party_names.csv")
    if validation_records:
        pd.DataFrame(validation_records).to_csv(csv_validation, index=False)
        logger.success("Saved scientific_validations.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the PDF processing pipeline.")
    parser.add_argument(
        "input_folder", type=str, help="Path to the folder containing PDF files."
    )
    args = parser.parse_args()

    process_pdfs(args.input_folder)
