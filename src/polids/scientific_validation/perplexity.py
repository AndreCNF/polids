from typing import Any, Dict, List, Tuple
import json
import backoff
from pydantic import ValidationError
import requests  # type: ignore[import]

from polids.config import settings
from polids.scientific_validation.base import (
    ScientificValidation,
    ScientificValidator,
)

# Define the overall behavior and constraints for the Perplexity AI model
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
ALLOWED_SOURCES = [
    # --- General Knowledge ---
    # Sources offering broad, encyclopedic information.
    "wikipedia.org",  # Crowd-sourced general knowledge encyclopedia
    # --- Data Aggregators ---
    # Platforms specializing in collecting, analyzing, and visualizing data on various topics.
    "ourworldindata.org",  # Accessible global data visualization & analysis
    # --- International Organizations ---
    # Official websites of major international bodies providing data, reports, and policy guidelines.
    "oecd.org",  # Organisation for Economic Co-operation and Development data & reports
    "un.org",  # United Nations reports & policy guidelines (global issues)
    "worldbank.org",  # World Bank global economic & development data
    # --- Research & Policy Analysis Institutes ---
    # Organizations focused on specific research areas, often influencing policy.
    "nber.org",  # National Bureau of Economic Research (influential economics)
    # --- Research Aggregators & Databases ---
    # Platforms providing access to collections of academic research papers.
    "core.ac.uk",  # Aggregator for open access research papers (multidisciplinary)
    "ncbi.nlm.nih.gov",  # National Center for Biotechnology Information (biomedical literature)
    "arxiv.org",  # Open access preprints (physics, math, CS, quantitative biology, etc.)
    "sci-hub.box",  # Tool for accessing paywalled scientific papers (legality varies)
]
MODEL_NAME = "sonar-pro"  # Model name for Perplexity AI; chosen for its alignment and detailed reasoning
CONTEXT_SIZE = "high"  # Context size for the search; "high" for comprehensive results with more sources


def extract_valid_json(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts and returns only the valid JSON part from a response object.

    This function assumes that the response has a structure where the valid JSON
    is included in the 'content' field of the first choice's message, after the
    closing "</think>" marker. Any markdown code fences (e.g. ```json) are stripped.

    Parameters:
        response (dict): The full API response object.

    Returns:
        dict: The parsed JSON object extracted from the content.

    Raises:
        ValueError: If no valid JSON can be parsed from the content.
    """
    # Navigate to the 'content' field; adjust if your structure differs.
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

    # Find the index of the closing </think> tag.
    marker = "</think>"
    idx = content.rfind(marker)

    if idx == -1:
        # If marker not found, try parsing the entire content.
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                "No </think> marker found and content is not valid JSON"
            ) from e

    # Extract the substring after the marker.
    json_str = content[idx + len(marker) :].strip()

    # Remove markdown code fence markers if present.
    if json_str.startswith("```json"):
        json_str = json_str[len("```json") :].strip()
    if json_str.startswith("```"):
        json_str = json_str[3:].strip()
    if json_str.endswith("```"):
        json_str = json_str[:-3].strip()

    try:
        parsed_json = json.loads(json_str)
        return parsed_json
    except json.JSONDecodeError as e:
        raise ValueError("Failed to parse valid JSON from response content") from e


@backoff.on_exception(
    backoff.expo,
    ValidationError,
    max_tries=5,
    max_time=60,
)
def search_on_perplexity(
    policy: str,
    model_name: str,
    search_context_size: str | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    perplexity_api_key: str = settings.perplexity_api_key,  # type: ignore[assignment]
    allowed_sources: list | None = None,
) -> tuple[ScientificValidation, list]:
    """
    Search for scientific validation of a policy proposal using Perplexity AI.

    Args:
        policy (str): The policy proposal to validate.
        model_name (str): The model name to use for the search.
        search_context_size (str, optional): The size of the search context (low, medium, high).
        system_prompt (str): The system prompt for the model.
        perplexity_api_key (str): The API key for Perplexity AI.
        allowed_sources (list, optional): A list of allowed sources for the search.

    Returns:
        tuple[ScientificValidation, list]: A tuple containing the validation result and citations.
    """
    request_payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": policy},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"schema": ScientificValidation.model_json_schema()},
        },
    }

    if allowed_sources:
        # Only search on the allowed sources
        request_payload["search_domain_filter"] = allowed_sources

    if search_context_size:
        # Define how many sources to use for the search
        assert search_context_size in ["low", "medium", "high"], (
            f"Invalid search context size: {search_context_size}. "
            "Must be one of: low, medium, high."
        )
        request_payload["web_search_options"] = {
            "search_context_size": search_context_size
        }

    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers={"Authorization": f"Bearer {perplexity_api_key}"},
        json=request_payload,
    ).json()

    citations = response.get("citations", [])
    response_content = response["choices"][0]["message"]["content"]

    if ("reasoning" in model_name) or ("<think>" in response_content):
        # Extract the valid JSON part from the response content
        json_content = extract_valid_json(response)
        # Parse the JSON content into the ScientificValidation model
        parsed_content = ScientificValidation.model_validate(json_content)
    else:
        # Parse the string content into the ScientificValidation model
        parsed_content = ScientificValidation.model_validate_json(response_content)

    return parsed_content, citations


class PerplexityScientificValidator(ScientificValidator):
    def __init__(
        self,
        perplexity_api_key: str = settings.perplexity_api_key,  # type: ignore[assignment]
        model_name: str = MODEL_NAME,
        context_size: str = CONTEXT_SIZE,
        system_prompt: str = SYSTEM_PROMPT,
        allowed_sources: list | None = ALLOWED_SOURCES,
    ):
        """
        Initialize the PerplexityScientificValidator with Perplexity AI parameters.

        Args:
            perplexity_api_key (str): The API key for Perplexity AI.
            model_name (str): The model name to use for the search.
            context_size (str): The size of the search context (low, medium, high).
            system_prompt (str): The system prompt for the model.
            allowed_sources (list, optional): A list of allowed sources for the search.
        """
        self.perplexity_api_key = perplexity_api_key
        self.model_name = model_name
        self.context_size = context_size
        self.system_prompt = system_prompt
        self.allowed_sources = allowed_sources

    def process(self, policy_proposal: str) -> Tuple[ScientificValidation, List[Any]]:
        """
        Process a policy proposal and return a structured validation result.

        Args:
            policy_proposal (str): The policy proposal to validate.

        Returns:
            ScientificValidation: A structured validation result containing the analysis of the policy proposal.
            List[Any]: A list of sources or evidence used in the validation process.
        """
        validation_result, citations = search_on_perplexity(
            policy=policy_proposal,
            model_name=self.model_name,
            search_context_size=self.context_size,
            system_prompt=self.system_prompt,
            perplexity_api_key=self.perplexity_api_key,
            allowed_sources=self.allowed_sources,
        )
        return validation_result, citations
