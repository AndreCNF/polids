<div align="center">

![polids](https://github.com/AndreCNF/polids/blob/main/data/polids_logo.png?raw=true)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/andrecnf/polids/main/app/app.py)

Analysis of political data and output of it through apps.

</div>

## Data

This project has its foundation on the electoral manifestos in markdown form, which are retrieved from [Política Para Todos](https://github.com/Politica-Para-Todos/manifestos). I highly recommend exploring their [GitHub](https://github.com/Politica-Para-Todos) and their [page](https://www.politicaparatodos.pt/) for other projects and a better understanding of Portuguese political data.

## Word cloud

![word_cloud](https://github.com/AndreCNF/polids/blob/main/data/portugal_2022/word_clouds/all_parties.png?raw=true)

A common way to visualise the content of a text is through a word cloud, a visual representation of the most frequent words in a text. Beyond just displaying the most frequent words, I do some pre-processing, including:
* Removing punctuation
* Removing stop words
* Removing words with less than 3 characters
* Keeping nouns, adjectives, verbs and interjections

The style for this type of word cloud was obtained from this [article](https://towardsdatascience.com/how-to-make-word-clouds-in-python-that-dont-suck-86518cdcb61f).

## Topic presence

Politics tends to mainly address a set of core topics, for which each party defines their priority and intentions. In order to better understand their stance and how it correlates to our own, I created a simple regex process to estimate the percentage of sentences in the parties' manifesto that address each of the following topics:
* Climate
* Economy
* Education
* Health
* Infrastructure
* Science
* Social causes
* Politics and ideology
* Technology and entrepreneurship

You can see which words are defined for each topic [here](https://github.com/AndreCNF/polids/blob/main/data/portugal_2022/topics.yml).

## Sentiment analysis

Forming partnerships and setting coalitions is ever more crucial in modern politics. This might be harder to achieve if a party has profoundly negative and critical communication. In order to assess each party's "negativity", I've applied a [sentiment analysis model](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) to their manifestos, getting the percentage of sentences that are:
* Neutral
* Positive
* Negative

## Rationality VS Intuition

A [recent study](https://phys.org/news/2022-01-rationality-declined-decades.html) explored the decline of rationalism in favor of intuition over the last decades. A potential implication of this is that politics might become less scientific and rational, ending up more emotional and manipulative. So, inspired by the paper, I've created a simple regex process to estimate the percentage of sentences in the parties' manifesto that are:
* Rational
* Intuitive

You can see which words are defined for each approach [here](https://github.com/AndreCNF/polids/blob/main/data/portugal_2022/approaches.yml).

## Hate speech

The rise of populism and extremism often comes along with a rise in hate speech. As such, I've thought that it would be interesting to see what hate speech exists in each party's manifesto. For this task, I've used an [hate speech detection model from HuggingFace Models](https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-portugese).

## Parliament proposals and votes

[Coming soon...](https://twitter.com/andrecnferreira/status/1676151049273442304?s=20) 🔜
