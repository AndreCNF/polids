from polids.config import settings
from polids.topic_unification.base import (
    UnifiedTopicsOutput,
    TopicUnifier,
)

if settings.langfuse.log_to_langfuse:
    # If Langfuse is enabled, use the Langfuse OpenAI client
    from langfuse.openai import OpenAI  # type: ignore[import]
else:
    from openai import OpenAI


DEFAULT_LLM_NAME = "gpt-4.1-2025-04-14"
SYSTEM_PROMPT = """# Role
You are a political data analyst specialized in categorizing policy areas into standardized, concise supercategories for downstream analysis.

# Objective
Process a list of political topics with frequency counts to:
1. Generate concise, high-level policy categories (unified topics).
2. Map each input topic to its corresponding unified category.
3. Output the results as structured JSON following a defined schema.

# Input Format
A list where each line contains a topic and its frequency, separated by |. The list is pre-sorted in descending order by frequency.
```markdown
- Topic String 1 | Frequency 1
- Topic String 2 | Frequency 2
...
- Topic String N | Frequency N
```

# Task
1. **Parse Input:** Extract the topic string and frequency integer from each line of the input markdown list. The topic string alone will be used as the key in the output mapping.
2. **Cluster Semantically:** Group input topics that share core concepts, address related issues, or belong to a clear broader policy domain. Use the descending frequency order as a strong signal for identifying core themes, but prioritize semantic coherence.
3. **Define Unified Topics:** For each cluster, create a single, concise name for the unified topic.
4. **Map Inputs:** Assign each input topic string to exactly one unified topic name.
5. **Generate Output:** Format the results according to the specified JSON schema.

# Guidelines
- **Conciseness:** Unified topic names must be 1-5 words. (e.g., "Economy", "Healthcare", "Climate Action").
- **Mutual Exclusivity:** Unified topics must be distinct and non-overlapping in scope. Each input topic must map to only one unified topic.
- **Comprehensiveness:** All input topics must be mapped to a unified topic.
- **Political Relevance:** Use standard political science or common policy terminology for unified topic names.
- **Balance:** Aim for a reasonable number of unified topics (typically 5-12) – avoid over-granularity or excessive consolidation.

# Example
## Input
- Environmental Protection | 20
- Economic Growth Initiatives | 18
- Immigration reform | 17
- Climate Change Policies | 16
- Healthcare for All | 15
- Job creation programs | 14
- Strengthen military | 13
- Universal Healthcare Coverage | 12
- Improve public schools | 11
- Sustainable Economic Development | 10
- Border security funding | 10
- Renewable energy subsidies | 9
- Lower prescription drug costs | 8
- Reduce corporate tax rate | 7
- Affordable higher education | 6
- Combat Global Warming | 5
- Foreign aid reform | 4
##Output
{
  "unified_topics": [
    "Climate Action",
    "Economy",
    "Immigration & Border Security",
    "Healthcare",
    "Education",
    "Defense & Foreign Policy"
  ],
  "topic_mapping": {
    "Environmental Protection": "Climate Action",
    "Economic Growth Initiatives": "Economy",
    "Immigration reform": "Immigration",
    "Climate Change Policies": "Climate Action",
    "Healthcare for All": "Healthcare",
    "Job creation programs": "Economy",
    "Strengthen military": "Defense & Foreign Policy",
    "Universal Healthcare Coverage": "Healthcare",
    "Improve public schools": "Education",
    "Sustainable Economic Development": "Economy",
    "Border security funding": "Immigration",
    "Renewable energy subsidies": "Climate Action",
    "Lower prescription drug costs": "Healthcare",
    "Reduce corporate tax rate": "Economy",
    "Affordable higher education": "Education",
    "Combat Global Warming": "Climate Action",
    "Foreign aid reform": "Defense & Foreign Policy"
  }
}"""


class OpenAITopicUnifier(TopicUnifier):
    """
    Topic unifier using OpenAI's API.
    """

    def __init__(
        self, llm_name: str = DEFAULT_LLM_NAME, system_prompt: str = SYSTEM_PROMPT
    ):
        super().__init__()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.llm_name = llm_name
        self.system_prompt = system_prompt

    def get_unified_topics(self, input_markdown: str) -> UnifiedTopicsOutput:
        """
        Gets unified topics from the input Markdown text using OpenAI's API.

        Args:
            input_markdown (str): The input Markdown text containing topics.

        Returns:
            UnifiedTopicsOutput: The unified topics output.
        """
        completion = self.client.beta.chat.completions.parse(
            model=self.llm_name,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": input_markdown,
                },
            ],
            response_format=UnifiedTopicsOutput,  # Specify the schema for the structured output
            temperature=0,  # Low temperature should lead to less hallucination
            seed=42,  # Fix the seed for reproducibility
        )
        topics_unification_results = completion.choices[0].message.parsed
        assert isinstance(topics_unification_results, UnifiedTopicsOutput), (
            "Output does not match the expected schema."
        )
        return topics_unification_results
