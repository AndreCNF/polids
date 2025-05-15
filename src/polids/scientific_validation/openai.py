"""
OpenAI-based scientific policy validation using GPT-4o with web search tools.
"""

from typing import Any, Literal
from loguru import logger
import openai

from polids.config import settings
from polids.utils.backoff import llm_backoff
from polids.scientific_validation.base import (
    ScientificValidation,
    ScientificValidator,
)

SYSTEM_PROMPT = """**Objective:** Analyze the provided policy proposal to determine its scientific validity based on credible, current evidence obtained through web search. Produce a structured JSON output adhering to the `ScientificValidation` schema.

**Core Task:** Evaluate the scientific backing for the policy proposal. Your analysis must focus on:

1.  **Evidence Assessment:**
    - Determine if the **balance of scientific evidence** supports or refutes the policy's likely effectiveness or impact. This directly informs the `is_policy_supported_by_scientific_evidence` field.
    - Critically evaluate the **degree of consensus**

2.  **Source Prioritization:**
    - **Highest Priority:** Peer-reviewed scientific studies (especially systematic reviews, meta-analyses, RCTs), reports from established scientific organizations and governmental research bodies.
    - **Lower Priority:** Reputable news articles reporting on scientific findings (use mainly for context or pointers to primary sources, verify claims against primary literature).
    - **Avoid:** Opinion pieces, anecdotal evidence, single studies contradicted by broader evidence, non-credible sources.

3.  **Reasoning Formulation (`validation_reasoning` field):**
    - Provide a detailed explanation justifying the boolean field values (`is_policy_supported_by_scientific_evidence`, `is_scientific_consensus_present`).
    - Summarize the key findings from the most credible sources.
    - Explicitly mention supporting *and* conflicting evidence if found.
    - Reference specific evidence or sources briefly (e.g., "Smith et al., 2021 study showed X," "IPCC report indicates Y").

**Constraints & Guidelines:**
- Base your analysis solely on information retrieved from the web search.
- Focus on the most current and relevant scientific data.
- If evidence is mixed or limited, clearly state this in the reasoning and set boolean flags accordingly (e.g., `is_scientific_consensus_present` would likely be `False`).
- Present the final output (the structured JSON) in English, regardless of the policy proposal's original language.
- Do not include conversational text before or after the JSON output. Just provide the JSON object matching the `ScientificValidation` schema."""

MODEL_NAME = "gpt-4o-search-preview"


class OpenAIScientificValidator(ScientificValidator):
    """
    ScientificValidator implementation using OpenAI GPT-4o with web search tools.
    """

    def __init__(
        self,
        openai_api_key: str = settings.openai_api_key,  # type: ignore[assignment]
        model_name: str = MODEL_NAME,
        system_prompt: str = SYSTEM_PROMPT,
        search_context_size: Literal["low", "medium", "high"] = "high",
        temperature: float | None = None,
        seed: int | None = None,
    ):
        """
        Initialize the OpenAIScientificValidator with OpenAI API parameters.

        Args:
            openai_api_key (str): The API key for OpenAI.
            model_name (str): The model name to use for the search.
            system_prompt (str): The system prompt for the model.
            search_context_size (Literal["low", "medium", "high"]): The size of the search context.
                This determines the number of sources to consider in the validation process.
                The default is "high", which means a more extensive search.
            temperature (float | None): The temperature parameter for controlling randomness in responses.
            seed (int | None): The seed parameter for random number generation, ensuring reproducibility.
        """
        self.client = openai.Client(api_key=openai_api_key)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.search_context_size = search_context_size
        self.temperature = temperature
        self.seed = seed

    @llm_backoff
    def process(
        self,
        policy_proposal: str,
    ) -> tuple[ScientificValidation, list[Any]]:
        """
        Process a policy proposal and return a structured validation result.

        Args:
            policy_proposal (str): The policy proposal to validate.

        Returns:
            ScientificValidation: A structured validation result containing the analysis of the policy proposal.
            list[Any]: A list of sources or evidence used in the validation process.
        """
        # Prepare optional kwargs for temperature (seed not supported by search-preview model)
        parse_kwargs: dict[str, float] = {}
        if self.temperature is not None:
            parse_kwargs["temperature"] = self.temperature
        completion = self.client.beta.chat.completions.parse(
            model=self.model_name,
            web_search_options={"search_context_size": self.search_context_size},
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": policy_proposal},
            ],
            response_format=ScientificValidation,
            **parse_kwargs,
        )
        parsed = completion.choices[0].message.parsed
        if not isinstance(parsed, ScientificValidation):
            raise ValueError(
                f"Schema validation failed: Expected ScientificValidation, but got {type(parsed)}. "
                f"Response content: {parsed}"
            )
        citations = completion.choices[0].message.annotations
        if not citations:
            logger.warning(
                f"No citations found in the OpenAI response. Policy: {policy_proposal}"
            )
            citations = []
        return parsed, citations
