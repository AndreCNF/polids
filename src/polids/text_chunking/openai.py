from typing import List
from tqdm.auto import tqdm  # type: ignore[import]

from polids.config import settings

if settings.langfuse.log_to_langfuse:
    # If Langfuse is enabled, use the Langfuse OpenAI client
    from langfuse.openai import OpenAI  # type: ignore[import]
else:
    from openai import OpenAI

from polids.utils import is_text_similar
from polids.text_chunking.base import SemanticChunksPerPage, TextChunker


class OpenAITextChunker(TextChunker):
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)

    def get_chunks(self, text_per_page: List[str]) -> List[SemanticChunksPerPage]:
        chunk_outputs: list[SemanticChunksPerPage] = []
        # Iterate through each page of the document, extracting semantic chunks
        for idx, current_page_text in tqdm(
            enumerate(text_per_page),
            total=len(text_per_page),
            desc="Processing pages",
        ):
            if idx == len(text_per_page) - 1:
                next_page_preview = ""
            else:
                # Provide the LLM with a tweet-sized preview of the next page to check if the last chunk is incomplete
                next_page_text = text_per_page[idx + 1]
                next_page_preview = next_page_text.split("\n\n")[0][:280]
            completion = self.client.beta.chat.completions.parse(
                # Using the mini version for cheaper processing; setting a specific version for reproducibility
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant specialized in semantic text chunking for Markdown documents. Your task is to divide the provided <current_page_text> into a list of semantically coherent chunks, preserving all original Markdown formatting exactly.",
                    },
                    {
                        "role": "user",
                        "content": f"""**Overall Goal:**
Analyze the <current_page_text> (which is in Markdown format from a political manifesto) and split it into semantically coherent chunks. **Each chunk should ideally represent a distinct policy proposal, argument, thematic topic, or logical section typical of such documents.** This chunking is intended to facilitate downstream NLP analysis for understanding policy positions and informing voters. Perform this task objectively based on text structure and topic shifts, regardless of the political viewpoint expressed. Use the <next_page_preview> **only** to determine if the very last chunk of the <current_page_text> is semantically incomplete because its specific topic or proposal clearly continues into the preview text.

**Analysis Process:**
1.  **Carefully Analyze:** First, read and fully understand the entire <current_page_text> and the <next_page_preview> to grasp the context, policy flow, and document structure.
2.  **Identify Semantic Breaks:** Determine logical breakpoints in the <current_page_text> based on shifts in **topic, introduction of new policy proposals, distinct arguments, or transitions between manifesto sections.**
3.  **Use Markdown & Document Cues:** Treat Markdown elements (headings `#`, `##`; lists `*`, `-`, `1.`; paragraphs separated by blank lines; thematic breaks `---`) **and common manifesto structures (e.g., explicitly numbered proposals, thematic chapters/sections)** as strong indicators for potential chunk boundaries, but always prioritize the semantic flow of policy ideas or arguments. A single chunk can span multiple paragraphs or elements if they detail the *same* core proposal or argument.
4.  **Check Final Chunk:** Evaluate if the *last* identified semantic unit (e.g., the end of a policy description) in <current_page_text> stops mid-thought and its specific topic clearly continues at the start of <next_page_preview>.
5.  **Format Output:** Construct the required structured output according to the provided schema.

**Critical Constraints:**
- **Exact Markdown Preservation:** You MUST NOT alter the <current_page_text>. Preserve ALL original Markdown syntax, whitespace (spaces, tabs, newlines), and characters exactly. The concatenation of the output `chunks` MUST perfectly reconstruct the original <current_page_text> string.
- **Language Agnostic:** Perform semantic analysis regardless of the text's language, preserving the original language within the chunks.

<current_page_text>
{current_page_text}
</current_page_text>
<next_page_preview>
{next_page_preview}
</next_page_preview>""",
                    },
                ],
                response_format=SemanticChunksPerPage,  # Specify the schema for the structured output
                temperature=0,  # Low temperature should lead to less hallucination
                seed=42,  # Fix the seed for reproducibility
            )
            chunks_output = completion.choices[0].message.parsed
            assert isinstance(chunks_output, SemanticChunksPerPage), (
                "Output does not match the expected schema."
            )
            assert is_text_similar(
                expected=current_page_text, actual="\n".join(chunks_output.chunks)
            ), "Output chunks do not reconstruct the original text."
            chunk_outputs.append(chunks_output)
        return chunk_outputs

    def merge_chunks(self, chunk_outputs: List[SemanticChunksPerPage]) -> List[str]:
        """
        Merges chunks across pages based on the 'last_chunk_incomplete' flag.

        Takes a list of SemanticChunksPerPage objects (one per page) and
        concatenates the last chunk of page 'i' with the first chunk of
        page 'i+1' if the flag `last_chunk_incomplete` for page 'i' is True.

        Args:
            chunk_outputs: A list where each item represents a page and contains
                           its chunks and the incompleteness flag for its last chunk.

        Returns:
            A flattened list of strings representing the final, merged semantic chunks
            in the correct order, preserving the original text and Markdown.
        """
        final_chunks: list[str] = []
        carry_over_chunk: str | None = None
        total_pages = len(chunk_outputs)

        for i, page_data in enumerate(chunk_outputs):
            # Make a copy to avoid modifying the input list's contents directly
            current_chunks: list[str] = list(page_data.chunks)
            is_last_incomplete: bool = page_data.last_chunk_incomplete

            # Skip pages that resulted in no chunks
            if not current_chunks:
                # If there was a carry-over, it means the previous page ended
                # mid-chunk, and this page is empty. The carry-over chunk
                # should likely stand alone as the last chunk of the previous semantic unit.
                if carry_over_chunk is not None:
                    final_chunks.append(carry_over_chunk)
                    carry_over_chunk = None  # Reset carry-over
                continue  # Skip processing for this empty page

            # 1. Handle carry-over from the *previous* page
            if carry_over_chunk is not None:
                # Prepend the carried-over text to the first chunk of the current page
                current_chunks[0] = carry_over_chunk + " " + current_chunks[0]
                carry_over_chunk = None  # Reset carry-over as it's now merged

            # 2. Check if the *current* page's last chunk needs carrying over
            # This happens if the flag is True AND it's not the very last page overall
            is_last_page_overall = i == total_pages - 1

            if is_last_incomplete and not is_last_page_overall:
                # The last chunk of this page is incomplete and needs merging later.
                # Set it aside as the next carry_over_chunk.
                carry_over_chunk = current_chunks[-1]
                # Add all chunks *except the last one* to the final list for now.
                final_chunks.extend(current_chunks[:-1])
            else:
                # This page's last chunk is complete, OR it's the last page overall.
                # Add all chunks from this page (potentially modified by step 1)
                # to the final list.
                final_chunks.extend(current_chunks)
        return final_chunks
