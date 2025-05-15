import pytest

from polids.scientific_validation.base import ScientificValidation  # type: ignore[import]
from polids.scientific_validation.openai import OpenAIScientificValidator  # type: ignore[import]

# Example policies for basic integration tests
POLICY_TEST_CASES = [
    ("carbon_tax", "Implementing a carbon tax to reduce greenhouse gas emissions."),
    (
        "immigration_crime",
        "Blocking immigration from countries with different cultural backgrounds to reduce crime rates.",
    ),
]


@pytest.fixture(scope="module")
def validator():
    """
    Initialize the OpenAIScientificValidator using OPENAI_API_KEY env var.
    """
    return OpenAIScientificValidator(
        search_context_size="low",
    )  # Set to "low" for faster tests


@pytest.mark.parametrize("policy_name, policy_text", POLICY_TEST_CASES)
def test_openai_scientific_validator_basic(validator, policy_name, policy_text):
    """
    Basic test for OpenAIScientificValidator: ensure output schema, types, and non-empty reasoning and citations.
    """
    result, citations = validator.process(policy_text)

    # Validate result type
    assert isinstance(result, ScientificValidation), (
        f"Result for '{policy_name}' is not a ScientificValidation instance."
    )

    # Validate boolean fields are of correct type
    assert isinstance(result.is_policy_supported_by_scientific_evidence, bool), (
        f"'is_policy_supported_by_scientific_evidence' is not bool for '{policy_name}'."
    )
    assert isinstance(result.is_scientific_consensus_present, bool), (
        f"'is_scientific_consensus_present' is not bool for '{policy_name}'."
    )

    # Validate reasoning is non-empty
    assert result.validation_reasoning and result.validation_reasoning.strip(), (
        f"Validation reasoning is empty for '{policy_name}'."
    )

    # Validate citations list and contents
    assert isinstance(citations, list), (
        f"Citations for '{policy_name}' should be a list."
    )
    assert citations, (
        f"No citations returned for '{policy_name}', expected at least one citation."
    )
