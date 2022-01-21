from typing import Dict, List
import spacy
from string import punctuation

SPACY_NLP = spacy.load("pt_core_news_lg")


def _add_sentence_to_list(sentence: str, sentences_list: List[str]):
    """
    Add a sentence to the list of sentences.

    Args:
        sentence (str):
            Sentence to be added.
        sentences (List[str]):
            List of sentences.
    """
    while sentence.startswith(" "):
        # remove leading space
        sentence = sentence[1:]
    if all(c in punctuation for c in sentence) or len(sentence) == 1:
        # skip sentences with only punctuation
        return
    sentences_list.append(sentence)


def get_sentences(text: str) -> List[str]:
    """
    Get sentences from a text.

    Args:
        text (str):
            Text to be processed.

    Returns:
        List[str]:
            List of sentences.
    """
    # get the paragraphs
    paragraphs = text.split("\n")
    paragraphs = [p for p in paragraphs if p != ""]
    # get the sentences from the paragraphs
    sentences = list()
    for paragraph in paragraphs:
        if paragraph.startswith("#"):
            _add_sentence_to_list(paragraph, sentences)
            continue
        prev_sentence_idx = 0
        for idx in range(len(paragraph)):
            if idx + 1 < len(paragraph):
                if (paragraph[idx] == "." and not paragraph[idx + 1].isdigit()) or (
                    paragraph[idx] in "!?"
                ):
                    sentence = paragraph[prev_sentence_idx : idx + 1]
                    _add_sentence_to_list(sentence, sentences)
                    prev_sentence_idx = idx + 1
            else:
                sentence = paragraph[prev_sentence_idx:]
                _add_sentence_to_list(sentence, sentences)
    return sentences


def get_words(text: str) -> List[str]:
    """
    Get every word in the text that isn't a stopword or punctuation,
    and that is either a noun, adjective, verb or interjection
    (based on the [universal POS tags](https://universaldependencies.org/u/pos/))

    Args:
        text (str):
            Text to be processed.

    Returns:
        List[str]:
            List of words.
    """
    doc = SPACY_NLP(text)
    words = [
        word.text.replace("\n", "").replace("*", "")  # remove new line and bold symbols
        for word in doc
        if not word.is_stop  # remove stopwords
        and not word.is_punct  # remove punctuation
        and (
            word.pos_ == "NOUN"  # noun
            or word.pos_ == "ADJ"  # adjective
            or word.pos_ == "VERB"  # verb
            or word.pos_ == "INTJ"  # interjection
            or word.pos_ == "X"  # other
        )
    ]
    # remove blank words
    words = [word for word in words if word != ""]
    return words


def get_topical_sentences(
    sentences: List[str], topics: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Get lists of sentences per topic, based on the presence of
    words that are a part of the topic.

    Args:
        sentences (List[str]):
            List of sentences to analyse.
        topics (Dict[str, List[str]]):
            Dictionary of words per topic.

    Returns:
        Dict[str, List[str]]:
            Dictionary of sentences per topic.
    """
    topical_sentences = dict()
    for topic in topics:
        topical_sentences[topic] = list()
    for sentence in sentences:
        for topic in topics:
            if any(topical_word in sentence.lower() for topical_word in topics[topic]):
                topical_sentences[topic].append(sentence)
    return topical_sentences
