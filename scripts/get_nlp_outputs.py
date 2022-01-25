import os
import argparse
from tqdm.auto import tqdm
import sys
import pandas as pd

sys.path.append("utils/")
from data_utils import DATA_DIR, load_yaml_file, load_markdown_file
from nlp_utils import get_sentiment, get_hate_speech
from app_utils import get_sentences_from_party

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="portugal_2022")
args = parser.parse_args()
# load data
party_data = load_yaml_file(os.path.join(DATA_DIR, args.data_name, "parties_data.yml"))
party_names = list(party_data.keys())
# get outputs for each party
dfs = list()
for party in tqdm(party_names, desc="Getting nlp outputs per party"):
    sentences, _ = get_sentences_from_party(party)
    sentiment_df = get_sentiment(sentences)
    hate_speech_df = get_hate_speech(sentences, sentiment_df)
    # merge sentiment and hate speech dataframes
    sentiment_df.rename(
        columns={"label": "sentiment_label", "score": "sentiment_score"}, inplace=True
    )
    hate_speech_df.rename(
        columns={"label": "hate_speech_label", "score": "hate_speech_score"},
        inplace=True,
    )
    df = pd.merge(
        sentiment_df,
        hate_speech_df,
        left_index=True,
        right_index=True,
        suffixes=[None, "_redundant"],
    )
    df.drop(columns=["sentence_redundant"], inplace=True)
    df["party"] = party
    dfs.append(df)
# join all dataframes
df = pd.concat(dfs)
df.to_csv(os.path.join(DATA_DIR, args.data_name, "nlp_outputs.csv"), sep=";")
print("Done!")
