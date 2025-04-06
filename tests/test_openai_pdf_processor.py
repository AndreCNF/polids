from pathlib import Path

from polids.utils import remove_formatting_from_text, is_text_similar  # type: ignore[import]
from polids.pdf_processing.openai import OpenAIPDFProcessor  # type: ignore[import]


def test_process_pdf():
    # Path to the test PDF file
    pdf_path = Path("tests/data/test_electoral_program.pdf")
    processor = OpenAIPDFProcessor()

    # Process the PDF
    result = processor.process(pdf_path)

    # Assertions
    assert len(result) > 0  # Ensure the output has at least one page
    assert len(result) == 5  # Check if the output has the correct number of pages
    # Check text similarity using multiple methods
    # For the shorter text blocks
    assert remove_formatting_from_text(
        "programa eleitoral legislativas 2022"
    ) in remove_formatting_from_text(result[0]), (
        f"First page content '{result[0]}' doesn't contain expected text 'programa eleitoral legislativas 2022'"
    )
    assert is_text_similar(
        """Liberdade
Esquerda
Europa
Ecologia
versão
Programa aprovado
no Xi Congresso
de dia 12 de dezembro
dezembro
de 2021""",
        result[4],
        difflib_threshold=0.7,
        semantic_threshold=0.7,
    ), "Fifth page content doesn't match expected text with sufficient similarity"
    assert is_text_similar(
        """Índice
igualdade, Justiça Social e liberdade
Trabalho, rendimento, Tempo e Proteção Social
Saúde
Educação
Conhecimento, Ciência e Ensino Superior
Cultura e Arte
Habitação e Espaço Público
Coesão Territorial, Transportes e Mobilidade
Emergência Climática e Energia
Economia Circular
Agricultura e Florestas
Conservação da Natureza e Biodiversidade
Bem-estar e Direitos dos Animais
Águas, rios e Oceanos
Justiça
Estado e instituições
Democracia
Prevenção e Combate à Corrupção
Soberania Digital
Portugal na Europa e no Mundo""",
        result[2],
        difflib_threshold=0.7,
        semantic_threshold=0.7,
    ), "Third page content doesn't match expected text with sufficient similarity"

    # For the longer text blocks
    assert is_text_similar(
        """Concretizar o Futuro Uma sociedade justa num planeta saudável
Este é o compromisso político do LIVRE
para a próxima legislatura, com a defesa
da justiça social e ambiental, com a igual-
dade e os direitos humanos, com a liber-
dade e com o futuro. A mudança de que
precisamos para trans-
formar Portugal, a Eu-
ropa e o mundo come-
ça com a mobilização
em torno de um novo
modelo de desenvol-
vimento para Portu-
gal, assente na quali-
ficação abrangente e
avançada da popula-
ção, em políticas que
garantem justiça in-
tergeracional, susten-
tabilidade ambiental e
igualdade social. Este projeto de futuro
tem uma visão ecologista, cosmopolita,
libertária e universalista que antecipa os
desafios do século XXI .
A alternativa é LIVRE — não receamos
compromissos políticos, e na Assem-
bleia da República procuraremos, sem-
pre, acordos amplos para concretizar
as mudanças necessárias para um país
mais justo e sustentável, e as políticas
transformadoras de que necessitamos
urgentemente.
As eleições legislativas de 2022 aconte-
cem num momento extremamente deli-
cado para o país e num contexto exigente
em que urge responder às devastadoras
consequências da pandemia. Impor-
ta proteger os mais afetados pela crise
económica, agravada pela crise emer-
gente de energia e matérias-primas, acu-
dir a um Serviço Nacional de Saúde em
pré-rutura, enfrentar a
crise ecológica e traçar
uma estratégia de de-
senvolvimento inclu-
sivo e sustentável no
médio e longo prazos
para Portugal. Esta exi-
gência tem de ter uma
resposta progressista
à altura.
A partir de 30 de janei-
ro a política progres-
sista na Assembleia
da República será LIVRE : com uma cultu-
ra de diálogo e de compromisso político
em defesa do bem público e comum, que
permitirá à esquerda lutar por conquistas
concretas e palpáveis e ambicionar uma
visão de futuro que nos dê esperança no
mundo que construímos para as novas
gerações.
O LIVRE luta por uma mobilização ambi-
ciosa, empática e humanista dos cida-
dãos para a transformação do nosso fu-
turo colectivo. Queremos trabalhar a partir
da Assembleia da República porque acre-
ditamos que não há igualdade sem ecolo-
gia e liberdade sem democracia. Nestas
legislativas a alternativa é LIVRE , o partido
português da Esquerda Verde.""",
        result[1],
    ), "Second page content doesn't match expected text with sufficient similarity"
    assert is_text_similar(
        """residentes não habituais, assim como a
promoção de uma maior fiscalização ao
investimento estrangeiro.
15 reforçar a exigência legislativa de
adequação de habitações utilizadas para
fins turísticos, nomeadamente o aloja-
mento local, na qual se deve diferenciar a
exploração profissional da dos pequenos
proprietários, estabelecer nos Planos Dire-
tores Municipais limites máximos de área
bruta de construção por freguesia desti-
nada a estabelecimentos hoteleiros; pro-
mover meios efetivos de controlo do Alo-
jamento Local não registado ou a operar
em condições ilegais, através da criação
de uma equipa especializada para o efeito.
16 reforçar a capacitação técnica do
Estado e a reorganização dos serviços
do Estado que trabalham sobre a habita-
ção. A alteração orgânica e regulamentar
do Instituto de Habitação e Reabilitação
Urbana implica mais recursos técnicos, fi-
nanceiros e administrativos, também para
garantir a aplicação efetiva das verbas do
Plano de Recuperação e Resiliência que
vai gerir. Necessidade de articulação das
várias escalas e serviços de governação
sobre o tema da habitação, sobretudo
entre políticas de desenvolvimento (so-
ciais, económicas, culturais), políticas do
espaço e território (instrumentos de ges-
tão territorial e planos e estratégias se-
toriais) e captação de fundos nacionais,
europeus e internacionais.""",
        result[3],
    ), "Fourth page content doesn't match expected text with sufficient similarity"
