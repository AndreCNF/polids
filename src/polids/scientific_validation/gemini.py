"""
Gemini-based scientific policy validation using PydanticAI with web search tools.
"""

import os
from typing import Any, Literal

from loguru import logger

from polids.config import settings
from polids.scientific_validation.base import (
    ScientificValidation,
    ScientificValidator,
)
from polids.utils.backoff import llm_backoff

if settings.langfuse.log_to_langfuse:
    from langfuse import get_client

    langfuse = get_client()
    if not langfuse.auth_check():
        logger.error(
            "Authentication to langfuse failed. Please check your credentials and host."
        )

SYSTEM_PROMPT = """**Objective:** Analyze the provided policy proposal to determine its scientific validity based on credible, current evidence obtained through web search. Produce a structured JSON output adhering to the `ScientificValidation` schema.

**Core Task:** Evaluate the scientific backing for the policy proposal. Your analysis must focus on:

1.  **Evidence Assessment:**
    - Determine if the **balance of scientific evidence** supports or refutes the policy's likely effectiveness or impact. This directly informs the `is_policy_supported_by_scientific_evidence` field.
    - Critically evaluate the **degree of consensus** among reliable scientific sources. Is there broad agreement, significant debate, or insufficient evidence? This directly informs the `is_scientific_consensus_present` field. Remember, `True` requires near-unanimous agreement among sources on the *validation outcome* (supported or not supported).

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

MODEL_NAME = "gemini-3-flash-preview"


class GeminiScientificValidator(ScientificValidator):
    """
    ScientificValidator implementation using PydanticAI + Gemini + web search.
    """

    def __init__(
        self,
        google_api_key: str | None = settings.google_api_key,  # type: ignore[assignment]
        model_name: str = MODEL_NAME,
        system_prompt: str = SYSTEM_PROMPT,
        search_context_size: Literal["low", "medium", "high"] = "high",
        thinking_level: Literal["low", "high"] = "high",
    ):
        """
        Initialize the GeminiScientificValidator with Google / Gemini parameters.

        Args:
            google_api_key (str | None): The API key for Gemini.
            model_name (str): The Gemini model name.
            system_prompt (str): The system prompt for the model.
            search_context_size (Literal["low", "medium", "high"]): Web search context size.
            thinking_level (Literal["low", "high"]): Gemini thinking level.
        """
        self.google_api_key = google_api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.search_context_size = search_context_size
        self.thinking_level = thinking_level
        self._agent = self._build_agent()

    def _build_agent(self) -> Any:
        """
        Build the PydanticAI agent lazily to keep imports local to this implementation.
        """
        try:
            from pydantic_ai import Agent, PromptedOutput, WebSearchTool
            from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
            from pydantic_ai.providers.google import GoogleProvider

            # Initialize Pydantic AI instrumentation
            Agent.instrument_all()
        except ImportError as exc:
            raise ImportError(
                "GeminiScientificValidator requires `pydantic-ai-slim[google]`."
            ) from exc

        if self.google_api_key:
            os.environ["GOOGLE_API_KEY"] = self.google_api_key

        model_settings = GoogleModelSettings(
            google_thinking_config={"thinking_level": self.thinking_level}
        )
        provider = GoogleProvider(api_key=self.google_api_key)
        model = GoogleModel(self.model_name, provider=provider)

        return Agent(
            model,
            model_settings=model_settings,
            output_type=PromptedOutput(ScientificValidation),
            system_prompt=self.system_prompt,
            builtin_tools=[WebSearchTool(search_context_size=self.search_context_size)],
        )

    @staticmethod
    def _extract_urls_from_object(obj: Any) -> list[str]:
        """
        Recursively extract URL-like strings from a nested object.
        """
        urls: list[str] = []
        if obj is None:
            return urls

        if hasattr(obj, "model_dump"):
            obj = obj.model_dump(mode="json")

        if isinstance(obj, str):
            if obj.startswith(("http://", "https://")):
                return [obj]
            return []

        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and key in {
                    "uri",
                    "url",
                    "source_uri",
                    "sourceUrl",
                    "link",
                }:
                    if value.startswith(("http://", "https://")):
                        urls.append(value)
                else:
                    urls.extend(
                        GeminiScientificValidator._extract_urls_from_object(value)
                    )
            return urls

        if isinstance(obj, list):
            for item in obj:
                urls.extend(GeminiScientificValidator._extract_urls_from_object(item))
            return urls

        return urls

    @classmethod
    def _extract_citations_from_result(cls, result: Any) -> list[str]:
        """
        Extract citations from PydanticAI Gemini web-search builtin tool returns.
        """
        citations: list[str] = []
        seen: set[str] = set()

        responses: list[Any] = []
        response = getattr(result, "response", None)
        if response is not None:
            responses.append(response)

        all_messages = getattr(result, "all_messages", None)
        if callable(all_messages):
            for message in all_messages():
                if getattr(message, "kind", None) == "response":
                    responses.append(message)

        for response_msg in responses:
            builtin_tool_calls = getattr(response_msg, "builtin_tool_calls", [])
            for call_part, return_part in builtin_tool_calls:
                if getattr(call_part, "tool_name", None) != "web_search":
                    continue
                for url in cls._extract_urls_from_object(
                    getattr(return_part, "content", None)
                ):
                    if url not in seen:
                        seen.add(url)
                        citations.append(url)

        return citations

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
        result = self._agent.run_sync(policy_proposal)
        validation = result.output

        if not isinstance(validation, ScientificValidation):
            raise ValueError(
                f"Schema validation failed: Expected ScientificValidation, but got {type(validation)}. "
                f"Response content: {validation}"
            )

        citations = self._extract_citations_from_result(result)
        if not citations:
            logger.warning(
                f"No citations found in the Gemini response. Policy: {policy_proposal}"
            )

        return validation, citations
