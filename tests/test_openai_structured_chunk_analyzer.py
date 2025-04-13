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
            """- Recuperar a identidade ocidental e a matriz cultural cristã;
- Repatriar imigrantes subsídio-dependentes, criminosos, ilegais e que sejam inadaptáveis em termos de cultura, costumes e comportamento;
- Cortar os apoios e subsídios (discriminação positiva) para as minorias étnicas;
- Travar o crescimento do Islão em Portugal e proibir a construção de novas mesquitas;
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
                    hate_speech=True,
                    reason="The text includes explicit hostile statements such as repatriating immigrants labeled as 'subsidy-dependent, criminals, illegals' and cuts subsidies for ethnic minorities; it also targets Islam and same-sex marriage, promoting discrimination and exclusion against identifiable groups.",
                    targeted_groups=[
                        "immigrants",
                        "ethnic minorities",
                        "Muslims",
                        "LGBTQ+",
                    ],
                ),
                political_compass=PoliticalCompass(
                    economic="right", social="authoritarian"
                ),
            ),
        ),
        (
            "**Criar uma taxa universal sobre o carbono**, no quadro de uma reforma fiscal ambiental...",
            ManifestoChunkAnalysis(
                policy_proposals=["Create a universal carbon tax."],
                sentiment="positive",
                topic="sustainability",
                hate_speech=HateSpeechDetection(
                    hate_speech=False,
                    reason="The text does not contain any hate speech or discriminatory content.",
                    targeted_groups=[],
                ),
                political_compass=PoliticalCompass(
                    economic="left", social="libertarian"
                ),
            ),
        ),
        (
            """### SIMPLIFICAÇÃO E DESAGRAVAMENTO DO IRS COM INTRODUÇÃO DE TAXA ÚNICA DE 15%

- Implementação de uma taxa única de IRS de 15%, aplicada por igual a todos os rendimentos e para todos os contribuintes

- Isenção de IRS para rendimentos de trabalho até remuneração mensal de cerca de €664

- Isenção adicional de 200€ mensais por filho dependente e por progenitor (400€ em caso de famílias monoparentais)

- Eliminação de todas as deduções e benefícios fiscais em sede de IRS, com exceção das mencionadas no ponto anterior

- Transitoriamente, um sistema de duas taxas: 15% para rendimentos até 30.000€ e 28% no remanescente""",
            ManifestoChunkAnalysis(
                policy_proposals=[
                    "Implement a flat IRS rate of 15% for all income and taxpayers.",
                    "Exempt IRS for work income up to approximately €664 per month.",
                    "Additional exemption of €200 per month per dependent child and parent (or €400 for single-parent families).",
                    "Eliminate all deductions and tax benefits in IRS, except those mentioned above.",
                    "Temporarily, a two-rate system: 15% for income up to €30,000 and 28% on the remainder.",
                ],
                sentiment="neutral",
                topic="economy",
                hate_speech=HateSpeechDetection(
                    hate_speech=False,
                    reason="The text does not contain any hate speech or discriminatory content.",
                    targeted_groups=[],
                ),
                political_compass=PoliticalCompass(
                    economic="right", social="libertarian"
                ),
            ),
        ),
        (
            """# Identidade e Visão do Volt

Imagina-te numa situação onde tu e os teus concidadãos teriam de desenhar de raiz a sociedade em que viveriam. Que tipo de sociedade escolherias?""",
            ManifestoChunkAnalysis(
                policy_proposals=[],
                sentiment="neutral",
                topic="ideology",
                hate_speech=HateSpeechDetection(
                    hate_speech=False,
                    reason="The text does not contain any hate speech or discriminatory content.",
                    targeted_groups=[],
                ),
                political_compass=PoliticalCompass(
                    economic="center", social="libertarian"
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
    assert result.hate_speech.hate_speech == expected.hate_speech.hate_speech, (
        f"Hate speech mismatch: expected {expected.hate_speech}, got {result.hate_speech.hate_speech}"
    )
    assert (len(expected.hate_speech.targeted_groups) == 0) == (
        len(result.hate_speech.targeted_groups) == 0
    ), (
        f"Targeted hate speech groups don't match. Expected: {expected.hate_speech.targeted_groups}, got: {result.hate_speech.targeted_groups}"
    )

    # Check political compass
    assert result.political_compass.economic in ["left", "center", "right"], (
        f"Invalid economic compass value '{result.political_compass.economic}'. Expected one of 'left', 'center', or 'right'."
    )
    assert result.political_compass.social in [
        "libertarian",
        "center",
        "authoritarian",
    ], (
        f"Invalid social compass value '{result.political_compass.social}'. Expected one of 'libertarian', 'center', or 'authoritarian'."
    )
