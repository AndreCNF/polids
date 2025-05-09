import pandas as pd  # type: ignore[import]
from pydantic import BaseModel


def explode_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode list columns in a DataFrame into separate rows.

    Args:
        df (pd.DataFrame): DataFrame with list columns to explode.

    Returns:
        pd.DataFrame: DataFrame with exploded list columns.
    """
    list_columns = df.select_dtypes(include=["object"]).map(
        lambda x: isinstance(x, list)
    )
    if not list_columns.any().any():
        # No list typed columns to explode, return the DataFrame as it is
        return df
    else:
        # Iterate over each column that contains lists
        for col in list_columns.columns[list_columns.any()]:
            # Explode the list into separate rows
            df = df.explode(col)
        # Reset the index to ensure a clean DataFrame
        df.reset_index(drop=True, inplace=True)
        return df


def expand_dict_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand dictionary columns in a DataFrame into separate columns.

    Args:
        df (pd.DataFrame): DataFrame with dictionary columns to expand.

    Returns:
        pd.DataFrame: DataFrame with expanded dictionary columns.
    """
    dict_columns = df.select_dtypes(include=["object"]).map(
        lambda x: isinstance(x, dict)
    )
    if not dict_columns.any().any():
        # No dictionary typed columns to expand, return the DataFrame as it is
        return df
    else:
        # Iterate over each column that contains dictionaries
        for col in dict_columns.columns[dict_columns.any()]:
            # Expand the dictionary into columns
            expanded_dict_df = pd.json_normalize(df[col])
            # Rename expanded columns as they can have dots if it was a nested dict
            expanded_dict_df.columns = [
                col.replace(".", "_") if "." in col else col
                for col in expanded_dict_df.columns
            ]
            # Rename the columns to contain the original column name
            # as a prefix, followed by the sub-column name
            expanded_dict_df.columns = [
                f"{col}_{sub_col}" for sub_col in expanded_dict_df.columns
            ]
            # Concatenate the expanded DataFrame with the original DataFrame
            # and drop the original column
            df = pd.concat([df.drop(columns=[col]), expanded_dict_df], axis=1)
        # Rerun the function recursively to handle any new dictionary columns
        return expand_dict_columns(df)


def convert_pydantic_to_dataframe(pydantic_list: list[BaseModel]) -> pd.DataFrame:
    """
    Convert a list of Pydantic models to a pandas DataFrame.

    Args:
        pydantic_list (list[BaseModel]): List of Pydantic models to convert.
            Note that the input list should be instances of the one single
            Pydantic model class, not a list of different models.

    Returns:
        pd.DataFrame: DataFrame representation of the Pydantic models.

    Raises:
        AssertionError: If the input list is empty or contains different Pydantic model classes.
    """
    assert len(pydantic_list) > 0, "The input list must not be empty."
    assert all(isinstance(s, type(pydantic_list[0])) for s in pydantic_list), (
        "All elements in the list must be instances of the same Pydantic model class."
    )
    # Convert each Pydantic model to a dictionary to add to a DataFrame
    df = pd.DataFrame(s.model_dump() for s in pydantic_list)
    # Expand dictionary columns, if any
    df = expand_dict_columns(df)
    return df
