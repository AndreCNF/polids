from pathlib import Path

import pandas as pd

from polids.pipeline_runner import process_pdfs, run_rate_limited_tasks  # type: ignore[import]
from polids.scientific_validation.base import ScientificValidation  # type: ignore[import]
from polids.structured_analysis.base import (  # type: ignore[import]
    HateSpeechDetection,
    ManifestoChunkAnalysis,
    PoliticalCompass,
)
from polids.topic_unification.base import UnifiedTopic, UnifiedTopicsOutput  # type: ignore[import]


class _FakeRateLimitError(Exception):
    def __init__(self):
        super().__init__("status_code: 429 RESOURCE_EXHAUSTED")
        self.status_code = 429
        self.body = {"error": {"status": "RESOURCE_EXHAUSTED"}}


def test_run_rate_limited_tasks_collects_exhausted_failures(monkeypatch):
    import polids.pipeline_runner as pipeline_runner  # type: ignore[import]

    attempts: dict[str, int] = {}

    def worker(payload: str) -> str:
        attempts[payload] = attempts.get(payload, 0) + 1
        if payload == "fail":
            raise _FakeRateLimitError()
        return payload.upper()

    monkeypatch.setattr(pipeline_runner.time, "sleep", lambda _: None)

    result = run_rate_limited_tasks(
        tasks=[("ok", "ok"), ("fail", "fail")],
        worker_fn=worker,
        task_label="test",
        max_workers=2,
        max_retries=2,
        base_sleep_seconds=0.1,
        max_sleep_seconds=0.1,
        collect_exhausted_failures=True,
    )

    assert result.results == {"ok": "OK"}
    assert len(result.exhausted_failures) == 1
    assert attempts["ok"] == 1
    assert attempts["fail"] == 3
    assert result.exhausted_failures[0].task_id == "fail"
    assert result.exhausted_failures[0].retries_attempted == 2
    assert result.exhausted_failures[0].status_code == 429
    assert result.exhausted_failures[0].error_status == "RESOURCE_EXHAUSTED"


def test_process_pdfs_persists_validation_failures_and_continues(monkeypatch, tmp_path):
    import polids.pipeline_runner as pipeline_runner  # type: ignore[import]

    class _FakePDFProcessor:
        def process_batch(self, pdf_paths: list[Path]) -> list[list[str]]:
            return [["page content"] for _ in pdf_paths]

        def process(self, _: Path) -> list[str]:
            return ["page content"]

    class _FakeTextChunker:
        def process(self, _: list[str]) -> list[str]:
            return ["chunk content"]

    class _FakePartyName:
        full_name = "Example Party"
        short_name = "EP"
        is_confident = True

    class _FakePartyExtractor:
        def extract_party_names(self, _: list[str]) -> _FakePartyName:
            return _FakePartyName()

    class _FakeAnalyzer:
        def process(self, _: str) -> ManifestoChunkAnalysis:
            return ManifestoChunkAnalysis(
                policy_proposals=["proposal-ok", "proposal-fail"],
                sentiment="neutral",
                topic="economy",
                hate_speech=HateSpeechDetection(
                    is_hate_speech=False,
                    reason="",
                    targeted_groups=[],
                ),
                political_compass=PoliticalCompass(economic=0.0, social=0.0),
            )

    class _FakeValidator:
        attempts: dict[str, int] = {}

        def __init__(self, *args, **kwargs):
            del args, kwargs

        def process(self, proposal: str) -> tuple[ScientificValidation, list[str]]:
            self.attempts[proposal] = self.attempts.get(proposal, 0) + 1
            if proposal == "proposal-fail":
                raise _FakeRateLimitError()
            return (
                ScientificValidation(
                    is_policy_supported_by_scientific_evidence=True,
                    is_scientific_consensus_present=True,
                    validation_reasoning="supported",
                ),
                ["https://example.org/source"],
            )

    class _FakeTopicUnifier:
        def process(self, topic_counts: dict[str, int]) -> UnifiedTopicsOutput:
            return UnifiedTopicsOutput(
                unified_topics=[
                    UnifiedTopic(
                        name="Economy",
                        original_topics=list(topic_counts.keys()),
                    )
                ]
            )

    monkeypatch.setattr(pipeline_runner, "MistralPDFProcessor", _FakePDFProcessor)
    monkeypatch.setattr(pipeline_runner, "OpenAIPDFProcessor", _FakePDFProcessor)
    monkeypatch.setattr(pipeline_runner, "MarkerPDFProcessor", _FakePDFProcessor)
    monkeypatch.setattr(pipeline_runner, "OpenAITextChunker", _FakeTextChunker)
    monkeypatch.setattr(pipeline_runner, "MarkdownTextChunker", _FakeTextChunker)
    monkeypatch.setattr(
        pipeline_runner, "OpenAIPartyNameExtractor", _FakePartyExtractor
    )
    monkeypatch.setattr(pipeline_runner, "OpenAIStructuredChunkAnalyzer", _FakeAnalyzer)
    monkeypatch.setattr(pipeline_runner, "GeminiScientificValidator", _FakeValidator)
    monkeypatch.setattr(pipeline_runner, "OpenAITopicUnifier", _FakeTopicUnifier)
    monkeypatch.setattr(pipeline_runner.time, "sleep", lambda _: None)

    monkeypatch.setattr(pipeline_runner.settings, "llm_analysis_max_workers", 1)
    monkeypatch.setattr(pipeline_runner.settings, "llm_validation_max_workers", 1)
    monkeypatch.setattr(pipeline_runner.settings, "llm_rate_limit_max_retries", 2)
    monkeypatch.setattr(
        pipeline_runner.settings, "llm_rate_limit_base_sleep_seconds", 0.1
    )
    monkeypatch.setattr(
        pipeline_runner.settings, "llm_rate_limit_max_sleep_seconds", 0.1
    )
    monkeypatch.setattr(
        pipeline_runner.settings,
        "gemini_validation_model_name",
        "gemini-3-flash-preview",
    )
    monkeypatch.setattr(
        pipeline_runner.settings,
        "gemini_validation_search_context_size",
        "high",
    )
    monkeypatch.setattr(
        pipeline_runner.settings,
        "gemini_validation_thinking_level",
        "high",
    )

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "doc.pdf").write_bytes(b"%PDF-1.4 test")

    process_pdfs(str(input_dir))

    output_dir = input_dir / "output"
    validation_df = pd.read_csv(output_dir / "scientific_validations.csv")
    failures_df = pd.read_csv(output_dir / "scientific_validation_failures.csv")

    assert len(validation_df) == 1
    assert validation_df.loc[0, "proposal"] == "proposal-ok"

    expected_failure_columns = {
        "pdf_file",
        "chunk_index",
        "proposal_index",
        "proposal",
        "error_type",
        "status_code",
        "error_message",
        "retries_attempted",
        "timestamp_utc",
    }
    assert set(failures_df.columns) == expected_failure_columns
    assert len(failures_df) == 1
    assert failures_df.loc[0, "proposal"] == "proposal-fail"
    assert int(failures_df.loc[0, "retries_attempted"]) == 2
    assert int(failures_df.loc[0, "status_code"]) == 429
    assert _FakeValidator.attempts["proposal-fail"] == 3
