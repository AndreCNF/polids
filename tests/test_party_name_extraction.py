import pytest
from typing import Dict, List
from polids.utils.text_similarity import is_text_similar  # type: ignore[import]
from polids.party_name_extraction.openai import OpenAIPartyNameExtractor  # type: ignore[import]


@pytest.fixture
def chunks_by_party():
    return {
        "livre": [
            "# Programa Eleitoral Legislativas 2022 LIVRE",
            "## Concretizar o Futuro\n### Uma sociedade justa num planeta saudável\nEste é o compromisso político do LIVRE para a próxima legislatura, com a defesa da justiça social e ambiental, com a igualdade e os direitos humanos, com a liberdade e com o futuro. A mudança de que precisamos para transformar Portugal, a Europa e o mundo começa com a mobilização em torno de um novo modelo de desenvolvimento para Portugal, assente na qualificação abrangente e avançada da população, em políticas que garantem justiça intergeracional, sustentabilidade ambiental e igualdade social. Este projeto de futuro tem uma visão ecologista, cosmopolita, libertária e universalista que antecipa os desafios do século XXI.",
            "A alternativa é LIVRE — não receamos compromissos políticos, e na Assembleia da República procuraremos, sempre, acordos amplos para concretizar as mudanças necessárias para um país mais justo e sustentável, e as políticas transformadoras de que necessitamos urgentemente.",
            "As eleições legislativas de 2022 acontecem num momento extremamente delicado para o país e num contexto exigente em que urge responder às devastadoras consequências da pandemia. Importa proteger os mais afetados pela crise económica, agravada pela crise emergente de energia e matérias-primas, acudir a um Serviço Nacional de Saúde em pré-rutura, enfrentar a crise ecológica e traçar uma estratégia de desenvolvimento inclusivo e sustentável no médio e longo prazos para Portugal. Esta exigência tem de ter uma resposta progressista à altura.",
            "A partir de 30 de janeiro a política progressista na Assembleia da República será LIVRE: com uma cultura de diálogo e de compromisso político em defesa do bem público e comum, que permitirá à esquerda lutar por conquistas concretas e palpáveis e ambicionar uma visão de futuro que nos dê esperança no mundo que construímos para as novas gerações.",
            "O LIVRE luta por uma mobilização ambiciosa, empática e humanista dos cidadãos para a transformação do nosso futuro colectivo. Queremos trabalhar a partir da Assembleia da República porque acreditamos que não há igualdade sem ecologia e liberdade sem democracia. Nestas legislativas a alternativa é LIVRE, o partido português da Esquerda Verde.",
        ],
        "volt": [
            "## Identidade e Visão do Volt\n\nImagina-te numa situação onde tu e os teus concidadãos teriam de desenhar de raiz a sociedade em que viveriam. Que tipo de sociedade escolherias?\n\nO filósofo John Rawls propôs que, antes de escolher, cada um de nós colocasse um ‘véu da ignorância’. Ao colocar este véu, desconheceríamos o nosso ponto de partida ou as nossas circunstâncias na sociedade, ou seja, não saberíamos qual o nosso nível de riqueza, o nosso estatuto, a nossa orientação sexual, não saberíamos nada sobre o nosso lugar no mundo. Não possuindo estas informações, e procurando o nosso próprio bem-estar, seria certamente consensual a construção de uma sociedade em que ter uma vida digna é um direito real e universal, onde a liberdade de qualquer pessoa é apenas limitada pelo direito à liberdade do nosso próximo e onde todos têm o mesmo nível de oportunidades para prosperar e alcançar a sua própria definição de felicidade.",
            "Esta é a **visão** do Volt de uma sociedade justa, de um Portugal e de uma Europa para todos, e esta funciona como ponto de partida para a ideologia que guia a nossa ação política.",
            "Somos **progressistas** porque não nos resignamos ao estado atual das coisas, opondo-nos ao conservadorismo e procurando promover soluções inovadoras para os nossos problemas comuns e para o aperfeiçoamento da condição humana em sociedade, soluções onde prevaleçam os valores da liberdade, da igualdade de oportunidades, da solidariedade, da dignidade humana e dos direitos humanos. Além de social, o progressismo que defendemos é um que inclui o **ecologismo** porque vemos o bem-estar do planeta e dos outros seres vivos como um fim em si mesmo e como uma condição necessária para o bem-estar da sociedade. Por isto mesmo, acreditamos que a necessidade de sustentabilidade impõe limitações importantes à forma como a sociedade é conduzida e em todas as soluções que defendemos temos a preocupação de encontrar uma harmonia com o que nos rodeia e com o que nos sustenta vendo como prioridade a preservação do meio ambiente, a utilização responsável de recursos minerais e energéticos e o combate às alterações climáticas.",
            "Somos **pragmáticos** porque procuramos ser coerentes com a nossa visão e valores propondo e defendendo o que de facto funcione com base no consenso, no uso da razão e da evidência científica e nunca em dogmatismos ideológicos ou no egoísmo na procura do poder. Colocamo-nos ao centro do espetro político na forma como vemos o papel do Estado, pois vemos este como um meio e não como um fim, acreditamos que o Estado deve intervir tão pouco e tão rápido quanto possível e tanto e por quanto tempo for necessário.",
            "Somos **europeístas** porque, sendo Portugueses, sentimo-nos também Europeus. Vemos a União Europeia como parte da nossa identidade e futuro e, reconhecendo a interdependência do mundo em que vivemos, olhamos para o aprofundamento do projeto europeu e para a construção de uma federação europeia como o caminho para a resolução dos grandes desafios que enfrentamos. Queremos uma Europa unida e mais democrática que valoriza os seus cidadãos e os seus residentes, que não deixa ninguém para trás e que cria e mantém as condições necessárias para a realização do potencial único de cada um e da comunidade no seu todo. Uma Europa que se esforça continuamente para alcançar, em conjunto e de forma solidária, os mais elevados padrões de desenvolvimento humano, social, ambiental e técnico e que possa ajudar a guiar o resto do mundo através do exemplo na prossecução desses mesmos padrões, sendo um promotor chave da paz, da justiça, da prosperidade e da sustentabilidade global.",
        ],
    }


def test_extract_party_names(chunks_by_party: Dict[str, List[str]]):
    extractor = OpenAIPartyNameExtractor()
    for original_party_name, chunks in chunks_by_party.items():
        result = extractor.extract_party_names(chunks, batch_size=2)
        assert result.is_confident, (
            f"Extraction for {original_party_name} was not confident."
        )
        assert is_text_similar(
            expected=original_party_name, actual=result.full_name
        ) or is_text_similar(expected=original_party_name, actual=result.short_name), (
            f"Extracted name {result.full_name} ({result.short_name}) does not match expected {original_party_name}."
        )
