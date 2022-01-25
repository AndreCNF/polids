import yaml
import pandas as pd

DATA_DIR = "data"


def load_yaml_file(file_path):
    """
    Loads a yaml file and returns a dictionary with the contents.

    Args:
        file_path (str):
            Path to the yaml file

    Returns:
        yaml_dict:
            Dictionary with the contents of the yaml file
    """
    with open(file_path, "r") as stream:
        yaml_dict = yaml.safe_load(stream)
        return yaml_dict


def load_markdown_file(file_path):
    """
    Loads a markdown file and returns a string with the contents.

    Args:
        file_path (str):
            Path to the markdown file

    Returns:
        markdown_str:
            String with the contents of the markdown file
    """
    with open(file_path, "r") as stream:
        markdown_str = stream.read()
        return markdown_str


def get_counts(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """
    Get the labels count in a dataframe.

    Args:
        df (pd.DataFrame):
            Dataframe to get the counts from.
        label_col (str):
            Column name of the label column.

    Returns:
        counts_df:
            Dataframe with the counts.
    """
    count_df = df[label_col].value_counts().to_frame().reset_index()
    count_df.columns = [label_col, "sentence_count"]
    count_df["percent"] = count_df.sentence_count / count_df.sentence_count.sum() * 100
    return count_df
