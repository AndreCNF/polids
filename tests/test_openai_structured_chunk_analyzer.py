import pytest
from polids.structured_analysis.openai import OpenAIStructuredChunkAnalyzer  # type: ignore[import]
from polids.structured_analysis.base import (  # type: ignore[import]
    ManifestoChunkAnalysis,
    HateSpeechDetection,
    PoliticalCompass,
)
from polids.utils.text_similarity import is_text_similar  # type: ignore[import]


@pytest.fixture
def analyzer():
    return OpenAIStructuredChunkAnalyzer()


@pytest.mark.parametrize(
    "chunk_text, expected",
    [
        (
            """- Repatriar imigrantes subsídio-dependentes, criminosos, ilegais e que sejam inadaptáveis em termos de cultura, costumes e comportamento;
- Cortar os apoios e subsídios (discriminação positiva) para as minorias étnicas;
- Anular a lei do “casamento” entre pessoas do mesmo sexo;
- Limitar o acesso às forças armadas e demais forças de segurança só aos portugueses de raiz.""",
            ManifestoChunkAnalysis(
                policy_proposals=[
                    "Repatriate immigrants who depend on subsidies, are criminals, are illegal, and are culturally unadaptable.",
                    "Cut funding and subsidies for ethnic minorities.",
                    "Prohibit the construction of new mosques.",
                    "Nullify the law on same-sex marriage.",
                    "Limit access to the armed forces and other security forces exclusively to native Portuguese.",
                ],
                sentiment="negative",
                topic="migration",
                hate_speech=HateSpeechDetection(
                    is_hate_speech=True,
                    reason="The text includes explicit hostile statements such as repatriating immigrants labeled as 'subsidy-dependent, criminals, illegals' and cuts subsidies for ethnic minorities; it also targets Islam and same-sex marriage, promoting discrimination and exclusion against identifiable groups.",
                    targeted_groups=[
                        "immigrants",
                        "ethnic minorities",
                        "Muslims",
                        "LGBTQ+",
                    ],
                ),
                political_compass=PoliticalCompass(
                    economic=1.0,
                    social=1.0,  # right/authoritarian
                ),
            ),
        ),
        (
            """**Garantizar la renta básica universal y legalizar el matrimonio igualitario**.
Aumentar la inversión pública en educación y salud gratuitas para todos. Eliminar barreras legales para la identidad de género y orientación sexual. Promover campañas de inclusión y diversidad.""",
            ManifestoChunkAnalysis(
                policy_proposals=[
                    "Guarantee universal basic income.",
                    "Legalize same-sex marriage.",
                    "Increase public investment in free education and healthcare for all.",
                    "Eliminate legal barriers based on gender identity or sexual orientation.",
                    "Promote inclusion and diversity campaigns.",
                ],
                sentiment="positive",
                topic="rights",
                hate_speech=HateSpeechDetection(
                    is_hate_speech=False,
                    reason="",  # No reasoning if no hate speech
                    targeted_groups=[],
                ),
                political_compass=PoliticalCompass(
                    economic=-1.0,
                    social=-1.0,  # strongly left/libertarian
                ),
            ),
        ),
        (
            """# Identité et vision du parti
L'objectif est de créer une société équilibrée, où les libertés individuelles sont respectées, mais où il existe également des règles claires pour garantir le bien-être collectif.""",
            ManifestoChunkAnalysis(
                policy_proposals=[],
                sentiment="neutral",
                topic="ideology",
                hate_speech=HateSpeechDetection(
                    is_hate_speech=False,
                    reason="",  # No reasoning if no hate speech
                    targeted_groups=[],
                ),
                political_compass=PoliticalCompass(
                    economic=0.0,
                    social=0.0,  # center/center
                ),
            ),
        ),
    ],
)
def test_chunk_analysis(
    analyzer: OpenAIStructuredChunkAnalyzer,
    chunk_text: str,
    expected: ManifestoChunkAnalysis,
):
    result = analyzer.process(chunk_text)

    # Check if the result is an instance of ManifestoChunkAnalysis
    assert isinstance(result, ManifestoChunkAnalysis), (
        f"Output does not match the expected schema:\n{result}"
    )

    # Check policy proposals
    assert len(result.policy_proposals) <= len(expected.policy_proposals), (
        f"Found {len(result.policy_proposals)} proposals, expected at most {len(expected.policy_proposals)}."
    )
    for proposal in expected.policy_proposals:
        # Is the proposal concise, i.e. tweet-length?
        assert len(proposal) <= 280, (
            f"Proposal '{proposal}' isn't concise, exceeds tweet-length."
        )
        # Is the proposal similar to any of the extracted proposals?
        assert any(
            is_text_similar(
                expected=proposal,
                actual=extracted,
                difflib_threshold=0.8,
                semantic_threshold=0.7,
            )
            for extracted in result.policy_proposals
        ), (
            f"Expected proposal '{proposal}' not found in extracted proposals: {result.policy_proposals}"
        )

    # Check sentiment
    assert result.sentiment in ["positive", "neutral", "negative"], (
        f"Invalid sentiment value '{result.sentiment}'. Expected one of 'positive', 'neutral', or 'negative'."
    )

    # Check topic
    assert 1 <= len(result.topic.split()) <= 2, (
        f"Topic should be one or two words. Found: {result.topic}"
    )
    assert (
        "," not in result.topic and ";" not in result.topic and "-" not in result.topic
    ), (
        f"Topic should not contain separators like commas, semicolons, or hyphens. Found: {result.topic}"
    )

    # Check hate speech
    assert result.hate_speech.is_hate_speech == expected.hate_speech.is_hate_speech, (
        f"Hate speech mismatch: expected {expected.hate_speech}, got {result.hate_speech.is_hate_speech}"
    )
    if not expected.hate_speech.is_hate_speech:
        assert result.hate_speech.reason == "", (
            f"Expected empty reason for no hate speech, got: '{result.hate_speech.reason}'"
        )
    assert (len(expected.hate_speech.targeted_groups) == 0) == (
        len(result.hate_speech.targeted_groups) == 0
    ), (
        f"Targeted hate speech groups don't match. Expected: {expected.hate_speech.targeted_groups}, got: {result.hate_speech.targeted_groups}"
    )

    # Check political compass (floats, directionality)
    assert isinstance(result.political_compass.economic, float), (
        f"Economic compass value should be float, got {type(result.political_compass.economic)}"
    )
    assert -1.0 <= result.political_compass.economic <= 1.0, (
        f"Economic compass value {result.political_compass.economic} out of range [-1, 1]"
    )
    assert isinstance(result.political_compass.social, float), (
        f"Social compass value should be float, got {type(result.political_compass.social)}"
    )
    assert -1.0 <= result.political_compass.social <= 1.0, (
        f"Social compass value {result.political_compass.social} out of range [-1, 1]"
    )
    # Check directionality (sign) for economic
    if expected.political_compass.economic > 0:
        assert result.political_compass.economic > 0.1, (
            f"Expected economic compass to be positive (right), got {result.political_compass.economic}"
        )
    elif expected.political_compass.economic < 0:
        assert result.political_compass.economic < -0.1, (
            f"Expected economic compass to be negative (left), got {result.political_compass.economic}"
        )
    else:
        assert abs(result.political_compass.economic) < 0.2, (
            f"Expected economic compass to be near zero (center), got {result.political_compass.economic}"
        )
    # Check directionality (sign) for social
    if expected.political_compass.social > 0:
        assert result.political_compass.social > 0.1, (
            f"Expected social compass to be positive (authoritarian), got {result.political_compass.social}"
        )
    elif expected.political_compass.social < 0:
        assert result.political_compass.social < -0.1, (
            f"Expected social compass to be negative (libertarian), got {result.political_compass.social}"
        )
    else:
        assert abs(result.political_compass.social) < 0.2, (
            f"Expected social compass to be near zero (center), got {result.political_compass.social}"
        )
