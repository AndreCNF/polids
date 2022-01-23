import os
import argparse
from tqdm.auto import tqdm
import sys

sys.path.append("utils/")
from data_utils import DATA_DIR, load_yaml_file, load_markdown_file
from nlp_utils import get_words
from viz_utils import get_word_cloud

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="portugal_2022")
args = parser.parse_args()
# load data
party_data = load_yaml_file(os.path.join(DATA_DIR, args.data_name, "parties_data.yml"))
party_names = list(party_data.keys())
program_txt = {
    party: load_markdown_file(
        os.path.join(DATA_DIR, args.data_name, "programs", f"{party}.md")
    )
    for party in tqdm(party_names, desc="Loading programs")
}
words = {
    party: get_words(program_txt[party])
    for party in tqdm(party_names, desc="Getting words")
}
# get word cloud for each party
# for party in tqdm(party_names, desc="Getting word clouds"):
#     get_word_cloud(
#         words=words[party],
#         image_path=os.path.join(DATA_DIR, args.data_name, "word_clouds"),
#         image_name=f"{party}.png",
#     )
# get word cloud for all parties
print("Getting word cloud for all parties...")
all_words = [word for party in party_names for word in words[party]]
get_word_cloud(
    words=all_words,
    image_path=os.path.join(DATA_DIR, args.data_name, "word_clouds"),
    image_name="all_parties.png",
)
print("Done!")
