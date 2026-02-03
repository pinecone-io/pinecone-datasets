import numpy as np
import pandas as pd

from pinecone_datasets.dataset import Dataset
from pinecone_datasets.dataset_fsreader import DatasetFSReader
from pinecone_datasets.dataset_fswriter import DatasetFSWriter


def test_read_pandas_dataframe(tmpdir):
    d = [
        {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "sparse_values": {"indices": [1, 2, 3], "values": [0.1, 0.2, 0.3]},
            "metadata": {"title": "title1", "url": "url1"},
            "blob": None,
        },
        {
            "id": "2",
            "values": [0.4, 0.5, 0.6],
            "sparse_values": {"indices": [4, 5, 6], "values": [0.4, 0.5, 0.6]},
            "metadata": {"title": "title2", "url": "url2"},
            "blob": None,
        },
    ]
    df = pd.DataFrame(d)

    schema_documents = [
        ("id", False, None),
        ("values", False, None),
        ("sparse_values", True, None),
        ("metadata", True, None),
        ("blob", True, None),
    ]

    # create tempdir
    dataset_name = "test_read_pandas_dataframe"
    tmpdir.mkdir(dataset_name)

    read_df = Dataset._read_pandas_dataframe(
        df, column_mapping=None, schema=schema_documents
    )
    assert isinstance(read_df, pd.DataFrame)

    # check if the dataframe is the same
    pd.testing.assert_frame_equal(df, read_df)

    # test None case
    none_df = Dataset._read_pandas_dataframe(
        None, column_mapping=None, schema=schema_documents
    )
    assert none_df.empty

    for k, _, _ in schema_documents:
        assert k in read_df.columns
        assert k in none_df.columns


def test_convert_metadata_from_dict_to_json():
    d1 = {"a": 1, "b": 2}
    s1 = '{"a": 1, "b": 2}'
    assert DatasetFSWriter._convert_metadata_from_dict_to_json(d1) == s1
    assert (
        DatasetFSReader._convert_metadata_from_json_to_dict(
            DatasetFSWriter._convert_metadata_from_dict_to_json(d1)
        )
        == d1
    )

    d2 = {"a": 1, "b": None}
    s2 = '{"a": 1, "b": null}'
    assert DatasetFSWriter._convert_metadata_from_dict_to_json(d2) == s2
    assert (
        DatasetFSReader._convert_metadata_from_json_to_dict(
            DatasetFSWriter._convert_metadata_from_dict_to_json(d2)
        )
        == d2
    )

    d3 = None
    s3 = None
    assert DatasetFSWriter._convert_metadata_from_dict_to_json(d3) == s3
    assert (
        DatasetFSReader._convert_metadata_from_json_to_dict(
            DatasetFSWriter._convert_metadata_from_dict_to_json(d3)
        )
        == d3
    )

    d4 = {"a": 1, "b": np.nan}
    s4 = '{"a": 1, "b": NaN}'
    assert DatasetFSWriter._convert_metadata_from_dict_to_json(d4) == s4

    # TODO WTF?
    # print({"a": 1, "b": np.nan})
    # print(Dataset._convert_metadata_from_json_to_dict(Dataset._convert_metadata_from_dict_to_json(d4)))
    # print(type(Dataset._convert_metadata_from_json_to_dict(Dataset._convert_metadata_from_dict_to_json(d4))['b']))
    # print(type(np.nan))
    # assert Dataset._convert_metadata_from_json_to_dict(Dataset._convert_metadata_from_dict_to_json(d4)) == d4
