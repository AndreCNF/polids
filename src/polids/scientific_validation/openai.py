"""OpenAI-based scientific policy validation using PydanticAI with web search tools."""

import os
import re
from typing import Any, Literal

from loguru import logger

from polids.config import settings
from polids.scientific_validation.base import (
    ScientificValidation,
    ScientificValidator,
)

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

MODEL_NAME = "gpt-5.2-2025-12-11"


class OpenAIScientificValidator(ScientificValidator):
    """
    ScientificValidator implementation using PydanticAI + OpenAI Responses + web search.
    """

    def __init__(
        self,
        openai_api_key: str | None = settings.openai_api_key,  # type: ignore[assignment]
        model_name: str = MODEL_NAME,
        system_prompt: str = SYSTEM_PROMPT,
        search_context_size: Literal["low", "medium", "high"] = "high",
        reasoning_effort: Literal["low", "medium", "high"] | None = "low",
        reasoning_summary: Literal["auto", "concise", "detailed"] | None = None,
        text_verbosity: Literal["low", "medium", "high"] | None = "low",
        temperature: float | None = None,
        top_p: float | None = None,
    ):
        """
        Initialize the OpenAIScientificValidator with OpenAI / Responses parameters.

        Args:
            openai_api_key (str | None): API key for OpenAI.
            model_name (str): OpenAI model name.
            system_prompt (str): The system prompt for the model.
            search_context_size (Literal["low", "medium", "high"]): Web search context size.
            reasoning_effort (Literal["low", "medium", "high"] | None): Optional reasoning effort.
            reasoning_summary (Literal["auto", "concise", "detailed"] | None): Optional reasoning summary verbosity.
            text_verbosity (Literal["low", "medium", "high"] | None): Optional final text verbosity for Responses API.
            temperature (float | None): Optional sampling temperature.
            top_p (float | None): Optional nucleus sampling.
        """
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.search_context_size = search_context_size
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.text_verbosity = text_verbosity
        self.temperature = temperature
        self.top_p = top_p
        self._agent = self._build_agent()

    def _build_agent(self) -> Any:
        """
        Build the PydanticAI agent lazily to keep imports local to this implementation.
        """
        try:
            from pydantic_ai import Agent, WebSearchTool
            from pydantic_ai.models.openai import (
                OpenAIResponsesModel,
                OpenAIResponsesModelSettings,
            )
            from pydantic_ai.providers.openai import OpenAIProvider

            # Initialize Pydantic AI instrumentation
            Agent.instrument_all()
        except ImportError as exc:
            raise ImportError(
                "OpenAIScientificValidator requires `pydantic-ai-slim[openai]`."
            ) from exc

        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key

        model_settings: OpenAIResponsesModelSettings = {
            # Include source payloads from the built-in web_search tool.
            "openai_include_web_search_sources": True,
        }
        if self.reasoning_effort is not None:
            model_settings["openai_reasoning_effort"] = self.reasoning_effort
        if self.reasoning_summary is not None:
            model_settings["openai_reasoning_summary"] = self.reasoning_summary
        if self.text_verbosity is not None:
            model_settings["openai_text_verbosity"] = self.text_verbosity
        if self.temperature is not None:
            model_settings["temperature"] = self.temperature
        if self.top_p is not None:
            model_settings["top_p"] = self.top_p

        provider = OpenAIProvider(api_key=self.openai_api_key)
        model = OpenAIResponsesModel(self.model_name, provider=provider)

        return Agent(
            model,
            model_settings=model_settings,
            output_type=ScientificValidation,
            system_prompt=self.system_prompt,
            builtin_tools=[
                WebSearchTool(
                    search_context_size=self.search_context_size,
                )
            ],
        )

    @staticmethod
    def _extract_urls_from_text(text: str) -> list[str]:
        """
        Extract URL-like substrings from free text, normalizing trailing punctuation.
        """
        pattern = re.compile(r"https?://[^\s<>\]\)\"']+")
        urls: list[str] = []
        for match in pattern.findall(text):
            normalized = match.rstrip(".,;:!?)]}>'\"")
            if normalized.startswith(("http://", "https://")):
                urls.append(normalized)
        return urls

    @staticmethod
    def _normalize_url(candidate: str) -> str | None:
        """Normalize URL-like strings and discard invalid values."""
        value = candidate.strip()
        if not value:
            return None
        if value.startswith(("http://", "https://")):
            return value
        if value.startswith("www."):
            return f"https://{value}"
        return None

    @staticmethod
    def _extract_source_descriptors_from_object(obj: Any) -> list[str]:
        """
        Extract source descriptors (URL/title/domain) from nested tool payloads.
        """
        sources: list[str] = []
        if obj is None:
            return sources

        if hasattr(obj, "model_dump"):
            obj = obj.model_dump(mode="json")

        if isinstance(obj, str):
            return OpenAIScientificValidator._extract_urls_from_text(obj)

        if isinstance(obj, dict):
            title: str | None = None
            domain: str | None = None
            url: str | None = None

            for key, value in obj.items():
                if not isinstance(value, str):
                    continue
                normalized = value.strip()
                if not normalized:
                    continue
                lowered_key = key.lower()
                if lowered_key in {"uri", "url", "source_uri", "retrieved_url", "link"}:
                    normalized_url = OpenAIScientificValidator._normalize_url(
                        normalized
                    )
                    if normalized_url:
                        url = normalized_url
                elif lowered_key in {"title", "source_title", "name"} and title is None:
                    title = normalized
                elif (
                    lowered_key in {"domain", "source_domain", "host"}
                    and domain is None
                ):
                    domain = normalized

            if url:
                sources.append(url)
            elif title and domain:
                sources.append(f"{title} ({domain})")
            elif title:
                sources.append(title)
            elif domain:
                sources.append(domain)

            for value in obj.values():
                sources.extend(
                    OpenAIScientificValidator._extract_source_descriptors_from_object(
                        value
                    )
                )
            return sources

        if isinstance(obj, list):
            for item in obj:
                sources.extend(
                    OpenAIScientificValidator._extract_source_descriptors_from_object(
                        item
                    )
                )
            return sources

        return sources

    @staticmethod
    def _extract_urls_from_object(obj: Any) -> list[str]:
        """
        Recursively extract URL-like strings from any nested object.
        """
        urls: list[str] = []
        if obj is None:
            return urls

        if hasattr(obj, "model_dump"):
            obj = obj.model_dump(mode="json")

        if isinstance(obj, str):
            return OpenAIScientificValidator._extract_urls_from_text(obj)

        if isinstance(obj, dict):
            for value in obj.values():
                urls.extend(OpenAIScientificValidator._extract_urls_from_object(value))
            return urls

        if isinstance(obj, list):
            for item in obj:
                urls.extend(OpenAIScientificValidator._extract_urls_from_object(item))
            return urls

        return urls

    @classmethod
    def _extract_citations_from_result(cls, result: Any) -> list[str]:
        """
        Extract citations from PydanticAI OpenAI tool returns and related metadata.
        """
        citations: list[str] = []
        seen: set[str] = set()

        def add_urls(urls: list[str]) -> None:
            for url in urls:
                if url not in seen:
                    seen.add(url)
                    citations.append(url)

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
            add_urls(
                cls._extract_urls_from_object(
                    getattr(response_msg, "provider_details", None)
                )
            )
            add_urls(
                cls._extract_urls_from_object(getattr(response_msg, "metadata", None))
            )

            parts = getattr(response_msg, "parts", [])
            for part in parts:
                part_kind = getattr(part, "part_kind", None)
                tool_name = getattr(part, "tool_name", None)

                if part_kind == "builtin-tool-call" and tool_name in {
                    "web_search",
                    "web_fetch",
                    "url_context",
                }:
                    add_urls(
                        cls._extract_source_descriptors_from_object(
                            getattr(part, "args", None)
                        )
                    )
                elif part_kind == "builtin-tool-return" and tool_name in {
                    "web_search",
                    "web_fetch",
                    "url_context",
                    "file_search",
                }:
                    add_urls(
                        cls._extract_source_descriptors_from_object(
                            getattr(part, "content", None)
                        )
                    )
                    add_urls(
                        cls._extract_urls_from_object(
                            getattr(part, "provider_details", None)
                        )
                    )
                else:
                    add_urls(
                        cls._extract_urls_from_object(
                            getattr(part, "provider_details", None)
                        )
                    )
                    add_urls(
                        cls._extract_urls_from_object(getattr(part, "content", None))
                    )

        output = getattr(result, "output", None)
        reasoning = getattr(output, "validation_reasoning", None)
        if isinstance(reasoning, str):
            add_urls(cls._extract_urls_from_text(reasoning))

        return citations

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
            response = getattr(result, "response", None)
            parts = getattr(response, "parts", []) if response is not None else []
            tool_parts = [
                getattr(part, "tool_name", None)
                for part in parts
                if getattr(part, "part_kind", None)
                in {"builtin-tool-call", "builtin-tool-return"}
            ]
            logger.warning(
                "No citations found in the OpenAI response. "
                f"Policy: {policy_proposal}. "
                f"Tool parts: {tool_parts}. "
                f"Finish reason: {getattr(response, 'finish_reason', None)}"
            )

        return validation, citations
