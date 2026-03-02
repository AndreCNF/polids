import pytest
from pydantic_ai.messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ModelResponse,
    TextPart,
)

from polids.scientific_validation.base import ScientificValidation  # type: ignore[import]
from polids.scientific_validation.gemini import GeminiScientificValidator  # type: ignore[import]


class _FakeRunResult:
    def __init__(self, output: ScientificValidation, response: ModelResponse):
        self.output = output
        self.response = response

    def all_messages(self):
        return [self.response]


class _FakeAgent:
    def __init__(self, result: _FakeRunResult):
        self._result = result

    def run_sync(self, _: str) -> _FakeRunResult:
        return self._result


class _FailingAgent:
    def __init__(self):
        self.calls = 0

    def run_sync(self, _: str) -> _FakeRunResult:
        self.calls += 1
        raise RuntimeError("boom")


def test_extract_citations_from_result_with_web_search_urls():
    response = ModelResponse(
        parts=[
            BuiltinToolCallPart(
                tool_name="web_search", tool_call_id="call-1", args={"queries": ["x"]}
            ),
            BuiltinToolReturnPart(
                tool_name="web_search",
                tool_call_id="call-1",
                content=[
                    {"title": "A", "uri": "https://example.org/a"},
                    {"title": "B", "uri": "https://example.org/b"},
                    {"title": "A duplicate", "uri": "https://example.org/a"},
                ],
            ),
            TextPart(content="final response"),
        ]
    )
    result = _FakeRunResult(
        output=ScientificValidation(
            is_policy_supported_by_scientific_evidence=True,
            is_scientific_consensus_present=True,
            validation_reasoning="Reasoning.",
        ),
        response=response,
    )

    citations = GeminiScientificValidator._extract_citations_from_result(result)
    assert citations == ["https://example.org/a", "https://example.org/b"]


def test_extract_citations_from_unpaired_web_search_return():
    response = ModelResponse(
        parts=[
            BuiltinToolReturnPart(
                tool_name="web_search",
                tool_call_id="call-1",
                content=[
                    {"title": "Only return", "uri": "https://example.org/unpaired"}
                ],
            ),
        ]
    )
    result = _FakeRunResult(
        output=ScientificValidation(
            is_policy_supported_by_scientific_evidence=True,
            is_scientific_consensus_present=False,
            validation_reasoning="Reasoning.",
        ),
        response=response,
    )

    citations = GeminiScientificValidator._extract_citations_from_result(result)
    assert citations == ["https://example.org/unpaired"]


def test_extract_citations_from_web_search_title_without_uri():
    response = ModelResponse(
        parts=[
            BuiltinToolReturnPart(
                tool_name="web_search",
                tool_call_id="call-1",
                content=[
                    {"title": "OECD report on housing affordability", "uri": None},
                    {"title": "CBS mobility dashboard", "domain": "cbs.nl"},
                ],
            ),
        ]
    )
    result = _FakeRunResult(
        output=ScientificValidation(
            is_policy_supported_by_scientific_evidence=True,
            is_scientific_consensus_present=False,
            validation_reasoning="Reasoning.",
        ),
        response=response,
    )

    citations = GeminiScientificValidator._extract_citations_from_result(result)
    assert citations == [
        "OECD report on housing affordability",
        "CBS mobility dashboard (cbs.nl)",
    ]


def test_extract_citations_from_web_fetch_return():
    response = ModelResponse(
        parts=[
            BuiltinToolCallPart(
                tool_name="web_fetch",
                tool_call_id="call-1",
                args={"urls": ["https://example.org/call-arg"]},
            ),
            BuiltinToolReturnPart(
                tool_name="web_fetch",
                tool_call_id="call-1",
                content=[{"retrieved_url": "https://example.org/fetched"}],
            ),
        ]
    )
    result = _FakeRunResult(
        output=ScientificValidation(
            is_policy_supported_by_scientific_evidence=True,
            is_scientific_consensus_present=False,
            validation_reasoning="Reasoning.",
        ),
        response=response,
    )

    citations = GeminiScientificValidator._extract_citations_from_result(result)
    assert citations == ["https://example.org/call-arg", "https://example.org/fetched"]


def test_process_returns_extracted_citations():
    response = ModelResponse(
        parts=[
            BuiltinToolCallPart(
                tool_name="web_search", tool_call_id="call-1", args={"queries": ["x"]}
            ),
            BuiltinToolReturnPart(
                tool_name="web_search",
                tool_call_id="call-1",
                content=[{"web": {"source_uri": "https://example.org/nested"}}],
            ),
        ]
    )
    expected = ScientificValidation(
        is_policy_supported_by_scientific_evidence=False,
        is_scientific_consensus_present=False,
        validation_reasoning="Mixed evidence.",
    )
    fake_result = _FakeRunResult(output=expected, response=response)

    validator = GeminiScientificValidator.__new__(GeminiScientificValidator)
    validator._agent = _FakeAgent(fake_result)

    parsed, citations = validator.process("policy")
    assert parsed == expected
    assert citations == ["https://example.org/nested"]


def test_process_falls_back_to_reasoning_urls():
    response = ModelResponse(parts=[TextPart(content="No tool parts here.")])
    expected = ScientificValidation(
        is_policy_supported_by_scientific_evidence=True,
        is_scientific_consensus_present=True,
        validation_reasoning=(
            "Summary based on sources: https://example.org/reasoning-1 and "
            "https://example.org/reasoning-2."
        ),
    )
    fake_result = _FakeRunResult(output=expected, response=response)

    validator = GeminiScientificValidator.__new__(GeminiScientificValidator)
    validator._agent = _FakeAgent(fake_result)

    parsed, citations = validator.process("policy")
    assert parsed == expected
    assert citations == [
        "https://example.org/reasoning-1",
        "https://example.org/reasoning-2",
    ]


def test_process_does_not_apply_internal_retries():
    validator = GeminiScientificValidator.__new__(GeminiScientificValidator)
    failing_agent = _FailingAgent()
    validator._agent = failing_agent

    with pytest.raises(RuntimeError, match="boom"):
        validator.process("policy")

    assert failing_agent.calls == 1
