from polids.config import settings
from polids.structured_analysis.base import (
    ManifestoChunkAnalysis,
    StructuredChunkAnalyzer,
)

if settings.langfuse.log_to_langfuse:
    # If Langfuse is enabled, use the Langfuse OpenAI client
    from langfuse.openai import OpenAI  # type: ignore[import]
else:
    from openai import OpenAI


class OpenAIStructuredChunkAnalyzer(StructuredChunkAnalyzer):
    def __init__(self):
        """
        Initializes the OpenAIStructuredChunkAnalyzer with an OpenAI client.
        """
        self.client = OpenAI(api_key=settings.openai_api_key)

    def process(self, chunk_text: str) -> ManifestoChunkAnalysis:
        """
        Processes a chunk of text from a political manifesto to extract structured analysis.

        Args:
            chunk_text (str): The text of the manifesto chunk to analyze.

        Returns:
            ManifestoChunkAnalysis: A structured analysis of the manifesto chunk.
        """
        completion = self.client.beta.chat.completions.parse(
            # Using the GPT 4.1 mini model that gets good enough output quality for cheap; setting a specific version for reproducibility
            model="gpt-4.1-mini-2025-04-14",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert political text analyst with deep knowledge of political ideologies, policy frameworks, and manifesto analysis. Your task is to analyze segments from political party electoral manifestos, focusing on precision, accuracy, and strict adherence to the provided schema.",
                },
                {
                    "role": "user",
                    "content": f"""**Analysis process**:
1. Policy Proposals:
- Carefully read the text to identify *concrete and specific* policy proposals.
- A valid policy proposal MUST:
  * Be a specific, actionable governmental commitment
  * Contain a clear, measurable action or initiative
  * Be concise (single short sentence), could fit in a tweet (280 characters)
- Do NOT include:
  * Vague ideological statements
  * General party values or principles without specific actions
  * Rhetorical flourishes or campaign slogans
  * Background information or contextual details
- Summarize each distinct proposal concisely, focusing *only* on the core action. Avoid including surrounding justification or filler text from the source. Aim for quality over quantity – extract only truly specific actions.
- If the text contains NO specific policy proposals, provide an empty list.
- Translate any non-English proposals into English.
- Provide the proposals as a list under field name: policy_proposals.

2. Sentiment:
- Evaluate only the *emotional tone* (not content or ideology).
- Select exactly ONE option: "positive" (optimistic, encouraging), "negative" (critical, pessimistic), or "neutral" (balanced, matter-of-fact).
- Provide this value under field name: sentiment.

3. Topic:
- Identify the *single, most dominant* political topic discussed in the chunk.
- Choose ONE specific keyword in English (avoid combining multiple topics).
- Common topics include but are NOT limited to: "economy", "healthcare", "education", "migration", "security", "environment", "foreign policy", "culture", "democracy", "justice".
- DO NOT use compound topics - select the most central focus only.
- Provide the chosen keyword under field name: topic.

4. Hate Speech Analysis:
- Analyze the text for hate speech based on the provided definition (hostility/prejudice/discrimination against groups based on characteristics).
- For field hate_speech:
  a. Set to true ONLY if the text contains explicit hostility, encourages discrimination, or promotes prejudice against identifiable groups.
  b. For subfield reason:
    - If hate speech is present: Quote SPECIFIC phrases and explain how they constitute hate speech.
    - If no hate speech: Briefly explain why the content was determined safe.
  c. For subfield targeted_groups: List ONLY groups explicitly targeted (if any) or leave empty.
- If hate_speech is false, targeted_groups should be empty.
- Place values under field name: hate_speech.

5. Political Compass Analysis:
- Assess political orientation with particular attention to explicit policy indicators:
  a. Economic:
    - "left": Strong indicators include support for wealth redistribution, expanded public services, market regulation, nationalization.
    - "right": Strong indicators include tax reduction, privatization, deregulation, free market emphasis, cutting government spending.
    - "center": Represents a mixed approach, balancing market principles with social welfare or targeted regulation (e.g., social market economy).
  b. Social:
    - "libertarian": Emphasizes personal freedoms, minimal state intervention in private life, civil liberties expansion.
    - "authoritarian": Emphasizes order, discipline, social conformity, expanded state powers over individuals.
    - "center": Balanced approach to personal freedoms and societal order.
- Provide selections under field name: political_compass with subfields economic and social.

**Task**:
Analyze the Markdown formatted text, applying the process described above.

**Input text**:
```markdown
{chunk_text}
```""",
                },
            ],
            response_format=ManifestoChunkAnalysis,  # Specify the schema for the structured output
            temperature=0,  # Low temperature should lead to less hallucination
            seed=42,  # Fix the seed for reproducibility
        )

        chunk_analysis = completion.choices[0].message.parsed
        assert isinstance(chunk_analysis, ManifestoChunkAnalysis), (
            "Output does not match the expected schema."
        )
        return chunk_analysis
