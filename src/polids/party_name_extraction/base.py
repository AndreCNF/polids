from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel, Field


class PartyName(BaseModel):
    """
    Represents the name of a political party, including its full and short forms.
    """

    full_name: str = Field(description="Full name of the party")
    short_name: str = Field(
        description="Short name of the party, abbreviated in an acronym"
    )
    is_confident: bool = Field(
        description="Whether the model is confident in its answer or not. "
        "If the party name is not in the text or it's ambiguous, this should be False."
    )


class PartyNameExtractor(ABC):
    @abstractmethod
    def extract_party_names(
        self, chunked_text: List[str], batch_size: int = 2
    ) -> PartyName:
        """
        Extracts a political party name from a list of pre-chunked text.

        Args:
            chunked_text (List[str]): A list of text chunks.
            batch_size (int): Number of chunks to process in each batch.

        Returns:
            PartyName: A PartyName object representing the extracted party name.
        """
        pass
