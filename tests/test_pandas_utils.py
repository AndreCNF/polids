import pytest
import pandas as pd  # type: ignore[import]
from pydantic import BaseModel

from polids.utils.pandas import (  # type: ignore[import]
    expand_dict_columns,
    convert_pydantic_to_dataframe,
    explode_list_columns,
)


def test_expand_dict_columns_with_valid_dict_columns():
    """Test expanding dictionary columns in a DataFrame."""
    df = pd.DataFrame(
        {"item_id": [1, 2], "data": [{"a": 10, "b": 20}, {"a": 30, "b": 40}]}
    )
    expanded_df = expand_dict_columns(df)
    assert "data_a" in expanded_df.columns, (
        f"Expected 'data_a' in columns, got {expanded_df.columns}"
    )
    assert "data_b" in expanded_df.columns, (
        f"Expected 'data_b' in columns, got {expanded_df.columns}"
    )
    assert expanded_df["data_a"].tolist() == [10, 30], (
        f"Expected [10, 30], got {expanded_df['data_a'].tolist()}"
    )
    assert expanded_df["data_b"].tolist() == [20, 40], (
        f"Expected [20, 40], got {expanded_df['data_b'].tolist()}"
    )


def test_expand_dict_columns_with_no_dict_columns():
    """Test DataFrame with no dictionary columns remains unchanged."""
    df = pd.DataFrame({"item_id": [1, 2], "value": [100, 200]})
    expanded_df = expand_dict_columns(df)
    assert df.equals(expanded_df), (
        f"Expected DataFrame to remain unchanged, got {expanded_df}"
    )


def test_expand_dict_columns_with_nested_dict_columns():
    """Test expanding nested dictionary columns."""
    df = pd.DataFrame({"item_id": [1], "data": [{"a": {"x": 1, "y": 2}, "b": 3}]})
    expanded_df = expand_dict_columns(df)
    assert "data_a_x" in expanded_df.columns, (
        f"Expected 'data_a_x' in columns, got {expanded_df.columns}"
    )
    assert "data_a_y" in expanded_df.columns, (
        f"Expected 'data_a_y' in columns, got {expanded_df.columns}"
    )
    assert "data_b" in expanded_df.columns, (
        f"Expected 'data_b' in columns, got {expanded_df.columns}"
    )
    assert expanded_df["data_a_x"].iloc[0] == 1, (
        f"Expected 1, got {expanded_df['data_a_x'].iloc[0]}"
    )
    assert expanded_df["data_a_y"].iloc[0] == 2, (
        f"Expected 2, got {expanded_df['data_a_y'].iloc[0]}"
    )
    assert expanded_df["data_b"].iloc[0] == 3, (
        f"Expected 3, got {expanded_df['data_b'].iloc[0]}"
    )


def test_explode_list_columns_with_valid_list_columns():
    """Test exploding list columns in a DataFrame."""
    df = pd.DataFrame({"id": [1, 2], "tags": [["tag1", "tag2"], ["tag3"]]})
    exploded_df = explode_list_columns(df)
    expected_df = pd.DataFrame({"id": [1, 1, 2], "tags": ["tag1", "tag2", "tag3"]})
    pd.testing.assert_frame_equal(exploded_df, expected_df, check_dtype=False)


def test_explode_list_columns_with_no_list_columns():
    """Test DataFrame with no list columns remains unchanged."""
    df = pd.DataFrame({"id": [1, 2], "value": [100, 200]})
    exploded_df = explode_list_columns(df)
    pd.testing.assert_frame_equal(df, exploded_df)


def test_explode_list_columns_with_empty_lists():
    """Test exploding list columns with empty lists."""
    df = pd.DataFrame({"id": [1, 2], "tags": [[], ["tag1", "tag2"]]})
    exploded_df = explode_list_columns(df)
    expected_df = pd.DataFrame({"id": [1, 2, 2], "tags": [None, "tag1", "tag2"]})
    pd.testing.assert_frame_equal(exploded_df, expected_df, check_dtype=False)


def test_explode_list_columns_with_mixed_types():
    """Test exploding list columns with mixed types."""
    df = pd.DataFrame({"id": [1, 2], "tags": [["tag1", 42], ["tag3"]]})
    exploded_df = explode_list_columns(df)
    expected_df = pd.DataFrame({"id": [1, 1, 2], "tags": ["tag1", 42, "tag3"]})
    pd.testing.assert_frame_equal(exploded_df, expected_df, check_dtype=False)


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    item_id: int
    name: str
    metadata: dict[str, int]


@pytest.fixture
def sample_pydantic_list():
    """Fixture to provide a list of SampleModel instances."""
    return [
        SampleModel(item_id=1, name="Alice", metadata={"age": 30, "score": 85}),
        SampleModel(item_id=2, name="Bob", metadata={"age": 25, "score": 90}),
    ]


def test_convert_pydantic_to_dataframe(sample_pydantic_list):
    """Test converting a list of Pydantic models to a DataFrame."""
    df = convert_pydantic_to_dataframe(sample_pydantic_list)
    assert "item_id" in df.columns, f"Expected 'id' in columns, got {df.columns}"
    assert "name" in df.columns, f"Expected 'name' in columns, got {df.columns}"
    assert "metadata_age" in df.columns, (
        f"Expected 'metadata_age' in columns, got {df.columns}"
    )
    assert "metadata_score" in df.columns, (
        f"Expected 'metadata_score' in columns, got {df.columns}"
    )
    assert df["item_id"].tolist() == [1, 2], f"Expected [1, 2], got {df['id'].tolist()}"
    assert df["name"].tolist() == ["Alice", "Bob"], (
        f"Expected ['Alice', 'Bob'], got {df['name'].tolist()}"
    )
    assert df["metadata_age"].tolist() == [30, 25], (
        f"Expected [30, 25], got {df['metadata_age'].tolist()}"
    )
    assert df["metadata_score"].tolist() == [85, 90], (
        f"Expected [85, 90], got {df['metadata_score'].tolist()}"
    )


def test_convert_pydantic_to_dataframe_empty_list():
    """Test converting an empty list raises an assertion error."""
    with pytest.raises(AssertionError, match="The input list must not be empty."):
        convert_pydantic_to_dataframe([])


def test_convert_pydantic_to_dataframe_mixed_models():
    """Test converting a list with mixed Pydantic models raises an assertion error."""

    class AnotherModel(BaseModel):
        item_id: int

    mixed_list = [
        SampleModel(item_id=1, name="Alice", metadata={"age": 30}),
        AnotherModel(item_id=2),
    ]
    with pytest.raises(
        AssertionError,
        match="All elements in the list must be instances of the same Pydantic model class.",
    ):
        convert_pydantic_to_dataframe(mixed_list)
