from abc import ABC, abstractmethod
import numpy as np
from pydantic import BaseModel, Field

from polids.utils.text_similarity import compute_text_similarity_scores


class UnifiedTopic(BaseModel):
    """
    Represents a high-level policy category with its constituent original topics.
    Ensures each unified topic contains at least two original topics.
    """

    name: str = Field(
        description="Concise name for the unified policy category (1-5 words)",
    )

    original_topics: list[str] = Field(
        description="List of original input topics mapped to this unified category",
    )


class UnifiedTopicsOutput(BaseModel):
    """
    Schema for political topic unification output with nested topic structure.
    Ensures mutual exclusivity through explicit grouping in separate unified topics.
    """

    unified_topics: list[UnifiedTopic] = Field(
        description="List of high-level policy categories with their constituent original topics. "
        "Each unified topic must contain at least two original topics, and all original "
        "topics from input must be included across the list. Categories must be mutually exclusive.",
    )


def convert_str_freq_to_markdown_list(
    str_freq: dict[str, int],
) -> str:
    """
    Converts a dictionary of string frequencies to a Markdown formatted list.

    Args:
        str_freq (dict[str, int]): A dictionary mapping strings to their frequencies.

    Returns:
        str: A Markdown formatted list of strings with their frequencies.
    """
    return "\n".join(
        f"- {key}: {value}"
        for key, value in sorted(str_freq.items(), key=lambda x: x[1], reverse=True)
    )


def map_topic(input_topic: str, topic_mapping: dict[str, list[str]]) -> str:
    """
    Maps the input topic to its corresponding unified topic using the provided mapping.
    If the input topic is not found in the mapping, it returns the most similar topic,
    based on character-based and semantic similarity.

    Args:
        input_topic (str): The topic to be mapped.
        topic_mapping (dict[str, list[str]]): A dictionary mapping output topics to input topics.

    Returns:
        str: The mapped topic or the most similar unified topic as a fallback.
    """
    mapped_output_topics = []
    for output_topic, input_topics in topic_mapping.items():
        for input_topic_ in input_topics:
            if input_topic == input_topic_:
                # Add the exact match to the list of mapped output topics
                # (note that we don't have a guarantee that there is only one exact match)
                mapped_output_topics.append(output_topic)
    if len(mapped_output_topics) == 1:
        return mapped_output_topics[0]
    elif len(mapped_output_topics) > 1:
        # Compare the mapped output topics to find the most similar one
        output_topics_to_compare = mapped_output_topics
    else:
        # Compare all output topics to find the most similar one
        output_topics_to_compare = list(topic_mapping.keys())
    output_topic_similarity_scores = {
        output_topic: np.mean(compute_text_similarity_scores(input_topic, output_topic))
        for output_topic in output_topics_to_compare
    }
    # Return the output topic with the highest similarity score
    return max(
        output_topic_similarity_scores,
        # Compare each dictionary key based on their average similarity score
        # and return the one with the highest score
        key=output_topic_similarity_scores.get,  # type: ignore
    )


def get_topic_mapping_from_unified_topics(
    unified_topics: UnifiedTopicsOutput,
) -> dict[str, list[str]]:
    """
    Converts the unified topics output into a mapping of unified topic names to their original topics.

    Args:
        unified_topics (UnifiedTopicsOutput): The unified topics output.

    Returns:
        dict[str, list[str]]: A dictionary mapping each unified topic name to its original topics.
    """
    return {ut.name: ut.original_topics for ut in unified_topics.unified_topics}


class TopicUnifier(ABC):
    def __init__(self):
        # Initialize the topic unifier with an empty mapping,
        # which will be populated with the unified topics and
        # their corresponding original topics.
        self.topic_mapping: dict[str, list[str]] = {}

    def convert_input_topic_frequencies_to_markdown_list(
        self, input_topic_frequencies: dict[str, int]
    ) -> str:
        """
        Converts the input topic frequencies to a Markdown formatted list.

        Args:
            input_topic_frequencies (dict[str, int]): A dictionary mapping input topics to their frequencies.

        Returns:
            str: A Markdown formatted list of input topics with their frequencies.
        """
        return convert_str_freq_to_markdown_list(input_topic_frequencies)

    @abstractmethod
    def get_unified_topics(self, input_markdown: str) -> UnifiedTopicsOutput:
        """
        Abstract method to get unified topics from the input Markdown text.

        Args:
            input_markdown (str): The input Markdown text containing topics.

        Returns:
            UnifiedTopicsOutput: The unified topics output.
        """
        pass

    def process(self, input_topic_frequencies: dict[str, int]) -> UnifiedTopicsOutput:
        """
        Processes the input topic frequencies to get unified topics.

        Args:
            input_topic_frequencies (dict[str, int]): A dictionary mapping input topics to their frequencies.

        Returns:
            UnifiedTopicsOutput: The unified topics output.
        """
        # Convert the input topic frequencies to a Markdown formatted list
        input_markdown = self.convert_input_topic_frequencies_to_markdown_list(
            input_topic_frequencies
        )
        # Get the unified topics from the input Markdown text
        unified_topics = self.get_unified_topics(input_markdown)
        # Update the topic mapping with the unified topics
        self.topic_mapping = get_topic_mapping_from_unified_topics(unified_topics)
        return unified_topics

    def map_input_topic_to_unified_topic(self, input_topic: str) -> str:
        """
        Maps the input topic to its corresponding unified topic.

        Args:
            input_topic (str): The topic to be mapped.

        Returns:
            str: The mapped unified topic.
        """
        return map_topic(input_topic=input_topic, topic_mapping=self.topic_mapping)
