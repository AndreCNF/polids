from abc import ABC, abstractmethod
from typing import Any, List, Tuple
from pydantic import BaseModel, Field


class ScientificValidation(BaseModel):
    """
    Model for validating a policy proposal based on scientific evidence.
    This model assesses whether a policy proposal is supported by
    scientific evidence and if there is a consensus among reliable sources.
    """

    is_policy_supported_by_scientific_evidence: bool = Field(
        description=(
            "Indicates whether the policy proposal is supported by scientific evidence from the searched sources. "
            "Set to `True` if the majority of reliable sources (e.g., peer-reviewed studies, reports from reputable organizations) "
            "provide evidence or arguments in favor of the policy's effectiveness or benefits. "
            "Set to `False` if the majority of sources oppose the policy, find it ineffective, or lack evidence to support it. "
            "Example: If a policy proposes a carbon tax to reduce emissions, set to `True` if most studies show carbon taxes reduce emissions."
        )
    )
    is_scientific_consensus_present: bool = Field(
        description=(
            "Indicates whether there is a clear consensus among reliable scientific sources regarding the policy's effectiveness or impact. "
            "Set to `True` ONLY if nearly all credible sources (e.g., peer-reviewed papers, expert analyses from trusted institutions) "
            "agree on whether the policy is supported or opposed (i.e., minimal conflicting evidence or opinions). "
            "Set to `False` if there is significant disagreement, mixed findings, or insufficient data among sources. "
            "Example: If 9 out of 10 studies agree a policy works, set to `True`. If only 6 out of 10 agree, set to `False`."
        )
    )
    validation_reasoning: str = Field(
        description=(
            "A detailed explanation of the validation outcome. Include: "
            "1. A summary of the key evidence or arguments from the sources regarding the policy's effectiveness or impact. "
            "2. Specific references to the sources (e.g., study titles, authors, faculties, organizations, etc) to support the conclusions. "
            "3. An explanation of why `is_policy_supported_by_scientific_evidence` and `is_scientific_consensus_present` were set to their respective values, "
            "including any conflicting evidence if present. "
            "Example: 'Most studies (e.g., Smith et al., 2020) support the policy due to evidence of reduced emissions by 20%, so `is_policy_supported_by_scientific_evidence` is `True`. "
            "However, two studies disagree on long-term effects, so `is_scientific_consensus_present` is `False`.'"
        )
    )


class ScientificValidator(ABC):
    @abstractmethod
    def process(self, policy_proposal: str) -> Tuple[ScientificValidation, List[Any]]:
        """
        Process a policy proposal and return a structured validation result.

        Args:
            policy_proposal (str): The policy proposal to validate.

        Returns:
            ScientificValidation: A structured validation result containing the analysis of the policy proposal.
            List[Any]: A list of sources or evidence used in the validation process.
        """
        pass
