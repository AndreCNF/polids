from typing import Dict, List
import spacy
from string import punctuation
import pandas as pd
from tqdm.auto import tqdm
from transformers import pipeline


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
    nlp = spacy.load("pt_core_news_lg")
    nlp.max_length = 8000000
    doc = nlp(text)
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
    # remove blank words and spaces
    words = [word for word in words if word != ""]
    words = [word.replace(" ", "") for word in words]
    # make all words lowercase
    words = [word.lower() for word in words]
    # remove undesired words
    words = [
        word
        for word in words
        if word not in ["se", "há", "política", "político", "políticos", "políticas"]
    ]
    # remove words with less than 3 characters
    words = [word for word in words if len(word) > 2]
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


def get_sentiment(sentences: List[str]) -> pd.DataFrame:
    """
    Get the sentiment of a list of sentences.

    Args:
        sentences (str):
            List of sentences to analyse.

    Returns:
        pd.DataFrame:
            Sentiment of the sentences.
    """
    sentiment_model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    sentiment_task = pipeline(
        "sentiment-analysis", model=sentiment_model_path, tokenizer=sentiment_model_path
    )
    sentiment_outputs = [
        sentiment_task(sentence)
        for sentence in tqdm(sentences, desc="Sentiment analysis")
    ]
    sentiments_dict = dict(label=[], score=[], sentence=[])
    for idx, output in enumerate(sentiment_outputs):
        sentiments_dict["label"].append(output[0]["label"])
        sentiments_dict["score"].append(output[0]["score"])
        sentiments_dict["sentence"].append(sentences[idx])
    sentiment_df = pd.DataFrame(sentiments_dict)
    sentiment_df["label"] = sentiment_df.label.map(
        dict(Positive="positivo", Negative="negativo", Neutral="neutro")
    )
    return sentiment_df


def get_hate_speech(sentences: List[str], sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the hate speech of a list of sentences.

    Args:
        sentences (str):
            List of sentences to analyse.
        sentiment_df (pd.DataFrame):
            Sentiment of the sentences.

    Returns:
        pd.DataFrame:
            Hate speech of the sentences.
    """
    hate_model_path = "Hate-speech-CNERG/dehatebert-mono-portugese"
    hate_task = pipeline(
        "text-classification", model=hate_model_path, tokenizer=hate_model_path
    )
    hate_outputs = [
        hate_task(sentence) for sentence in tqdm(sentences, desc="Hate speech analysis")
    ]
    hate_dict = dict(label=[], score=[], sentence=[])
    for idx, output in enumerate(hate_outputs):
        hate_dict["label"].append(output[0]["label"])
        hate_dict["score"].append(output[0]["score"])
        hate_dict["sentence"].append(sentences[idx])
    hate_df = pd.DataFrame(hate_dict)
    hate_df["label"] = hate_df.label.map(dict(HATE="ódio", NON_HATE="neutro"))
    hate_condition = (hate_df.label == "ódio") & (sentiment_df.label == "negativo")
    hate_df.loc[hate_condition, "label"] = "ódio"
    hate_df.loc[~hate_condition, "label"] = "neutro"
    return hate_df
