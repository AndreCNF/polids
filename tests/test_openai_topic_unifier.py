import pytest
from polids.topic_unification.openai import OpenAITopicUnifier  # type: ignore[import]
from polids.topic_unification.base import UnifiedTopicsOutput  # type: ignore[import]


@pytest.fixture
def topic_unifier() -> OpenAITopicUnifier:
    return OpenAITopicUnifier()


@pytest.fixture
def topics_to_unify() -> dict[str, int]:
    return {
        # Social Issues / Civil Rights
        "Protecting LGBTQ+ rights": 18,
        "Ensuring voting access for all": 25,
        "Criminal justice reform initiatives": 22,
        "Addressing systemic racism": 15,
        "Defending Second Amendment freedoms": 28,
        "Common-sense gun safety laws": 26,
        "Protecting freedom of religion": 12,
        "Police funding and accountability": 19,
        "Reproductive healthcare access": 23,
        # Technology & Infrastructure
        "Expanding rural broadband internet": 16,
        "Investing in national infrastructure (roads, bridges)": 30,
        "Cybersecurity for critical systems": 14,
        "Regulating big tech companies": 11,
        "Modernizing the power grid": 17,
        "Funding for scientific research (NIH, NSF)": 9,
        # Government Reform & Ethics
        "Campaign finance reform": 20,
        "Addressing political corruption": 13,
        "Term limits for elected officials": 8,
        "Protecting whistleblowers": 6,
        "Strengthening ethics regulations": 10,
        # Economy & Labor (Different Focus)
        "Raising the federal minimum wage": 24,
        "Supporting small business recovery": 21,
        "Protecting workers' right to organize": 18,
        "Addressing income inequality": 20,
        "Investing in workforce development programs": 15,
        # Agriculture & Environment (Different Focus)
        "Supporting sustainable farming practices": 7,
        "Ensuring food safety and security": 11,
        "Protecting national parks and public lands": 19,
        "Water resource management": 14,
        # Other
        "Addressing the opioid crisis": 22,
        "Improving veterans' healthcare and benefits": 27,
        "Affordable housing and homelessness": 16,
        "Disaster preparedness and relief funding": 10,
    }


def test_convert_input_topic_frequencies_to_markdown_list(
    topic_unifier: OpenAITopicUnifier, topics_to_unify: dict[str, int]
) -> None:
    expected_markdown = "\n".join(
        f"- {topic}: {freq}"
        for topic, freq in sorted(
            topics_to_unify.items(), key=lambda x: x[1], reverse=True
        )
    )
    result = topic_unifier.convert_input_topic_frequencies_to_markdown_list(
        topics_to_unify
    )
    assert result == expected_markdown, (
        f"Markdown conversion failed.\nExpected:\n{expected_markdown}\nGot:\n{result}"
    )


def test_get_unified_topics(
    topic_unifier: OpenAITopicUnifier, topics_to_unify: dict[str, int]
) -> None:
    input_markdown = topic_unifier.convert_input_topic_frequencies_to_markdown_list(
        topics_to_unify
    )
    result = topic_unifier.get_unified_topics(input_markdown)
    assert isinstance(result, UnifiedTopicsOutput), (
        f"Result is not of type UnifiedTopicsOutput.\nGot: {type(result)}"
    )
    assert len(result.unified_topics) > 0, (
        f"No unified topics were generated.\nGot: {result.unified_topics}"
    )
    unified_topic_names = [topic.name for topic in result.unified_topics]
    assert 5 <= len(unified_topic_names) <= 12, (
        f"Unexpected number of unified topics generated.\n"
        f"Expected: 5-12\nGot: {len(unified_topic_names)}"
    )
    assert all(1 <= len(topic.split()) <= 5 for topic in unified_topic_names), (
        f"Unified topic names are not concise (1-5 words).\nGot: {unified_topic_names}"
    )


def test_process(
    topic_unifier: OpenAITopicUnifier, topics_to_unify: dict[str, int]
) -> None:
    result = topic_unifier.process(topics_to_unify)
    assert isinstance(result, UnifiedTopicsOutput), (
        f"Result is not of type UnifiedTopicsOutput.\nGot: {type(result)}"
    )
    assert len(result.unified_topics) > 0, (
        f"No unified topics were generated.\nGot: {result.unified_topics}"
    )
    assert topic_unifier.topic_mapping, (
        f"Topic mapping was not populated.\nGot: {topic_unifier.topic_mapping}"
    )

    # Check for unmapped topics
    mapped_topics = [
        original_topic
        for unified_topic in result.unified_topics
        for original_topic in unified_topic.original_topics
    ]
    unmapped_topics = [
        topic for topic in topics_to_unify.keys() if topic not in mapped_topics
    ]
    assert not unmapped_topics, (
        f"Unmapped topics found.\nExpected all topics to be mapped.\n"
        f"Unmapped: {unmapped_topics}"
    )

    # Check for hallucinated topics
    hallucinated_topics = [
        topic for topic in mapped_topics if topic not in topics_to_unify.keys()
    ]
    assert not hallucinated_topics, (
        f"Hallucinated topics found.\nExpected no hallucinated topics.\n"
        f"Hallucinated: {hallucinated_topics}"
    )


def test_map_input_topic_to_unified_topic(
    topic_unifier: OpenAITopicUnifier, topics_to_unify: dict[str, int]
) -> None:
    topic_unifier.process(topics_to_unify)

    for input_topic in topics_to_unify.keys():
        result = topic_unifier.map_input_topic_to_unified_topic(input_topic)
        assert result in topic_unifier.topic_mapping, (
            f"Topic '{input_topic}' was not mapped correctly.\n"
            f"Expected one of: {list(topic_unifier.topic_mapping.keys())}\n"
            f"Got: {result}"
        )

    input_topic = "Nonexistent Topic"
    result = topic_unifier.map_input_topic_to_unified_topic(input_topic)
    assert result in topic_unifier.topic_mapping, (
        f"Nonexistent topic '{input_topic}' was not handled correctly.\n"
        f"Expected one of: {list(topic_unifier.topic_mapping.keys())}\n"
        f"Got: {result}"
    )
