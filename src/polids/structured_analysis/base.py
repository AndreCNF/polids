from abc import ABC, abstractmethod
from typing import List, Literal
from pydantic import BaseModel, Field


class HateSpeechDetection(BaseModel):
    """
    Model representing the analysis of hate speech within a text.
    """

    hate_speech: bool = Field(
        description=(
            "Boolean indicating if the text contains hate speech. Hate speech refers to any public expression "
            "that communicates hostility, animosity, or encourages violence, prejudice, discrimination, or "
            "intimidation against individuals or groups based on identifiable characteristics "
            "(e.g. race, ethnicity, national origin, religion, gender identity, sexual orientation, disability, age)."
        ),
    )
    reason: str = Field(
        description=(
            "Detailed explanation justifying the classification as hate speech or not. "
            "Include specific parts of the text that lead to this decision. If no hate speech is detected, explain why the content was determined safe."
        ),
    )
    targeted_groups: List[str] = Field(
        default_factory=list,
        description=(
            "List of groups or protected characteristics that are explicitly targeted by hate speech in the text. "
            "Examples include: 'race', 'religion', 'sexual orientation', 'gender identity', 'disability'. "
            "If no group is targeted, this list should be empty."
        ),
    )


class PoliticalCompass(BaseModel):
    """
    Model for the political compass analysis, representing two primary dimensions:
    - Economic: perspective on economic organization (left vs. right)
    - Social: perspective on personal freedom and state intervention (libertarian vs. authoritarian)
    """

    economic: Literal["left", "center", "right"] = Field(
        description=(
            "Economic stance on the political spectrum. 'left' suggests support for cooperative or state-driven "
            "economic models, 'right' indicates a free-market approach emphasizing individual competition, and "
            "'center' represents a moderate position combining elements of both views."
        ),
    )
    social: Literal["libertarian", "center", "authoritarian"] = Field(
        description=(
            "Social stance on the political spectrum. 'libertarian' represents maximal personal freedom with minimal state control, "
            "'authoritarian' indicates a preference for obedience to authority and strict social order, and "
            "'center' denotes a balanced or moderate view."
        ),
    )


class ManifestoChunkAnalysis(BaseModel):
    """
    Model for a comprehensive analysis of a segment (chunk) of a political party's electoral manifesto.
    Includes:
    - Policy proposals extraction
    - Sentiment analysis
    - Dominant political topic identification
    - Hate speech detection
    - Political compass positioning
    """

    policy_proposals: List[str] = Field(
        description=(
            "A list of one or more policy proposals extracted from the manifesto chunk. Each entry should be a concise, "
            "specific statement describing a proposed action to address a political issue. Proposals should be translated to English if needed."
        ),
    )
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description=(
            "The overall sentiment of the manifesto chunk. Valid values include 'positive', 'negative', or 'neutral', "
            "reflecting the emotional tone or attitude conveyed in the text."
        ),
    )
    topic: str = Field(
        description=(
            "The dominant political topic of the manifesto chunk. This should be captured in one or two keywords in English. "
            "Examples: 'economy', 'healthcare', 'education', 'migration', 'transport', 'science', 'sustainability', "
            "'welfare', 'social causes', 'ideology', 'infrastructure', 'business', 'technology', 'urban design'."
        ),
    )
    hate_speech: HateSpeechDetection = Field(
        description="Nested analysis of hate speech detection within the manifesto chunk.",
    )
    political_compass: PoliticalCompass = Field(
        description="Nested analysis mapping the manifesto chunk to positions on the political compass.",
    )


class StructuredChunkAnalyzer(ABC):
    @abstractmethod
    def process(self, chunk_text: str) -> ManifestoChunkAnalysis:
        """
        Processes a chunk of text from a political manifesto to extract structured analysis.

        Args:
            chunk_text (str): The text of the manifesto chunk to analyze.

        Returns:
            ManifestoChunkAnalysis: A structured analysis of the manifesto chunk.
        """
        pass
