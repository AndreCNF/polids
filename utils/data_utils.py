import yaml

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
