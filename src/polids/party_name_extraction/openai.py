from typing import List
from loguru import logger
from polids.party_name_extraction.base import PartyNameExtractor, PartyName

from polids.config import settings
from polids.utils.backoff import llm_backoff

if settings.langfuse.log_to_langfuse:
    # If Langfuse is enabled, use the Langfuse OpenAI client
    from langfuse.openai import OpenAI  # type: ignore[import]
else:
    from openai import OpenAI

SYSTEM_PROMPT = """# Role
You are a political-manifesto analyst specialized in identifying the authoring political party.

# Objective
Extract the party name from manifesto excerpts with high confidence.

# Instructions
1. Identify the party that authored the manifesto, not parties merely referenced in the text.
2. Return `is_confident=false` when identification is ambiguous, weakly supported, or absent.
3. Use `previous_guess` only as context to refine uncertain attempts.
4. Preserve the original language and naming as written in the text.
"""

USER_PROMPT_TEMPLATE = """<manifesto_text>
{manifesto_text}
</manifesto_text>

<previous_guess>
<full_name>{full_name}</full_name>
<short_name>{short_name}</short_name>
<is_confident>{is_confident}</is_confident>
</previous_guess>"""


class OpenAIPartyNameExtractor(PartyNameExtractor):
    def __init__(
        self,
        temperature: float | None = None,
        seed: int | None = None,
    ):
        """
        Initializes the OpenAIPartyNameExtractor with an OpenAI client.

        Args:
            temperature (float | None): Sampling temperature for chat completions.
            seed (int | None): Random seed for reproducibility.
        """
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.temperature = temperature
        self.seed = seed

    @llm_backoff
    def _call_openai_party_name_completion(
        self, current_chunks: list[str], previous_guess: PartyName
    ) -> PartyName:
        # Prepare optional kwargs for temperature and seed
        parse_kwargs: dict[str, float | int] = {}
        if self.temperature is not None:
            parse_kwargs["temperature"] = self.temperature
        if self.seed is not None:
            parse_kwargs["seed"] = self.seed
        manifesto_text = "\n\n".join(current_chunks)
        response = self.client.responses.parse(
            model="gpt-5-mini-2025-08-07",
            input=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(
                        manifesto_text=manifesto_text,
                        full_name=previous_guess.full_name,
                        short_name=previous_guess.short_name,
                        is_confident=previous_guess.is_confident,
                    ),
                },
            ],
            text_format=PartyName,
            **parse_kwargs,
        )
        return response.output_parsed

    def extract_party_names(
        self, chunked_text: List[str], batch_size: int = 2
    ) -> PartyName:
        """
        Extracts a political party name from a list of pre-chunked text.

        Args:
            chunked_text (List[str]): A list of text chunks.
            batch_size (int): Number of chunks to process in each batch.

        Returns:
            PartyName: A PartyName object representing the extracted party name.
        """
        is_confident = False
        idx = 0
        previous_guess = PartyName(full_name="", short_name="", is_confident=False)

        while not is_confident and idx < len(chunked_text):
            current_chunks = chunked_text[idx : idx + batch_size]
            party_name_guess = self._call_openai_party_name_completion(
                current_chunks=current_chunks, previous_guess=previous_guess
            )
            assert isinstance(party_name_guess, PartyName), (
                "Output does not match the expected schema."
            )

            is_confident = party_name_guess.is_confident
            if is_confident:
                return party_name_guess
            else:
                previous_guess = party_name_guess
                idx += batch_size

        logger.warning(
            "No confident party name found in the provided text chunks. Returning the last guess."
        )
        return party_name_guess
