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
