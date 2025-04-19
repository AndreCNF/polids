import pytest
from polids.scientific_validation.perplexity import (  # type: ignore[import]
    PerplexityScientificValidator,
    ALLOWED_SOURCES,
)
from polids.scientific_validation.base import ScientificValidation  # type: ignore[import]


@pytest.fixture
def validator():
    return PerplexityScientificValidator(
        context_size="low"  # Set to "low" for faster tests
    )


@pytest.mark.parametrize(
    "policy_name, policy_text, expected_validation",
    [
        (
            "carbon_tax",
            "Implementing a carbon tax to reduce greenhouse gas emissions.",
            ScientificValidation(
                is_policy_supported_by_scientific_evidence=True,
                is_scientific_consensus_present=True,
                validation_reasoning=(
                    "Most peer-reviewed studies and reports from reputable organizations "
                    "indicate that carbon taxes effectively reduce greenhouse gas emissions. "
                    "There is strong consensus among sources supporting this policy."
                ),
            ),
        ),
        (
            "vaccines",
            "Mandatory vaccination for all school-aged children to prevent outbreaks of infectious diseases.",
            ScientificValidation(
                is_policy_supported_by_scientific_evidence=True,
                is_scientific_consensus_present=True,
                validation_reasoning=(
                    "Scientific evidence overwhelmingly supports mandatory vaccination as an effective measure "
                    "to prevent outbreaks of infectious diseases. Studies show high efficacy and safety, "
                    "with broad consensus among experts."
                ),
            ),
        ),
        (
            "immigration_crime",
            "Blocking immigration from countries with different cultural backgrounds to reduce crime rates.",
            ScientificValidation(
                is_policy_supported_by_scientific_evidence=False,
                is_scientific_consensus_present=True,
                validation_reasoning=(
                    "Research indicates that cultural diversity does not correlate with increased crime rates. "
                    "Most studies suggest that crime is influenced by socioeconomic factors rather than cultural background. "
                    "There is a strong consensus among experts that this policy lacks scientific support."
                ),
            ),
        ),
    ],
)
def test_perplexity_scientific_validator(
    validator: PerplexityScientificValidator,
    policy_name: str,
    policy_text: str,
    expected_validation: ScientificValidation,
):
    result, citations = validator.process(policy_text)

    # Validate the result matches the expected output's booleans
    # (skipping the validation_reasoning as it could have high variance)
    assert (
        result.is_policy_supported_by_scientific_evidence
        == expected_validation.is_policy_supported_by_scientific_evidence
    ), (
        f"Mismatch in 'is_policy_supported_by_scientific_evidence' for policy '{policy_name}'. "
        f"Expected: {expected_validation.is_policy_supported_by_scientific_evidence}, "
        f"Got: {result.is_policy_supported_by_scientific_evidence}"
    )
    assert (
        result.is_scientific_consensus_present
        == expected_validation.is_scientific_consensus_present
    ), (
        f"Mismatch in 'is_scientific_consensus_present' for policy '{policy_name}'. "
        f"Expected: {expected_validation.is_scientific_consensus_present}, "
        f"Got: {result.is_scientific_consensus_present}"
    )

    # Validate that there is at least one citation
    assert len(citations) > 0, (
        f"No citations found for policy '{policy_name}'. Expected at least one citation."
    )

    # Validate that all citations are from allowed sources
    for citation in citations:
        assert any(source in citation for source in ALLOWED_SOURCES), (
            f"Citation '{citation}' is not from an allowed source. "
            f"Allowed sources are: {ALLOWED_SOURCES}"
        )

    # Validate that the validation reasoning is not empty
    assert len(result.validation_reasoning) > 0, (
        f"Validation reasoning is empty for policy '{policy_name}'. "
        "Expected a non-empty reasoning."
    )
