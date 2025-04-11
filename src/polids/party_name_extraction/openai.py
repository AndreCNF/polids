from typing import List
from loguru import logger
from polids.party_name_extraction.base import PartyNameExtractor, PartyName

from polids.config import settings

if settings.langfuse.log_to_langfuse:
    # If Langfuse is enabled, use the Langfuse OpenAI client
    from langfuse.openai import OpenAI  # type: ignore[import]
else:
    from openai import OpenAI


class OpenAIPartyNameExtractor(PartyNameExtractor):
    def __init__(self):
        """
        Initializes the OpenAIPartyNameExtractor with an OpenAI client.
        """
        self.client = OpenAI(api_key=settings.openai_api_key)

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
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant specialized in analyzing political manifestos to identify the name of the political party that authored the document. Your task is to extract the party's name from the provided <manifesto_text>, ensuring that the identification is confident and unambiguous. If the party name cannot be confidently identified, or if the text is vague, ambiguous, or mentions a party not affiliated with the manifesto, you must label the result as not confident (`is_confident = False`). Use the <previous_guess> to refer to the previous, non-confident guess of the party name as memory to refine your attempts.",
                    },
                    {
                        "role": "user",
                        "content": f"""**Overall Goal:**
Analyze the <manifesto_text> (which is in Markdown format from a political manifesto) and identify the name of the political party that authored it. **The identification must be confident and unambiguous.** If the text is vague, ambiguous, or mentions a party not affiliated with the manifesto, label the result as not confident (`is_confident = False`). Use the <previous_guess> to refer to the previous, non-confident guess of the party name as memory to refine your identification attempts.

**Analysis Process:**
1. **Carefully Analyze:** Read and fully understand the <manifesto_text> to grasp the context and identify any explicit mentions of the party's name.
2. **Check for Confidence:** Determine if the party's name is presented in a clear and unambiguous way. If the identification is not confident, label the result as `is_confident = False`.
3. **Use Previous Guesses:** If a previous, non-confident guess of the party name exists, use the <previous_guess> as memory to refine your identification attempts.
4. **Avoid False Positives:** Ensure that the identified name belongs to the party that authored the manifesto and not to another party mentioned in the text.
5. **Format Output:** Construct the required structured output according to the provided schema.

**Critical Constraints:**
- **Exact Markdown Preservation:** You MUST NOT alter the <manifesto_text>. Write the party name as it appears in the text, without any modifications.
- **Language Agnostic:** Preserve the original language as in the <manifesto_text> within the output.

<manifesto_text>
{current_chunks}
</manifesto_text>

<previous_guess>
    <full_name>{previous_guess.full_name}</full_name>
    <short_name>{previous_guess.short_name}</short_name>
    <is_confident>{previous_guess.is_confident}</is_confident>
</previous_guess>""",
                    },
                ],
                response_format=PartyName,
                temperature=0,
                seed=42,
            )
            party_name_guess = completion.choices[0].message.parsed
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
            "No confident party name found in the provided text chunks. "
            "Returning the last guess."
        )
        return party_name_guess
