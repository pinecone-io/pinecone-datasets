import pandas as pd
from pandas.testing import assert_frame_equal

from pinecone_datasets import Dataset, DatasetMetadata, DenseModelMetadata
from pinecone_datasets.ds_utils import transfer_keys_vectorized, import_documents_keys_from_blob_to_metadata
from pinecone_datasets.testing import assert_datasets_equal


df = pd.DataFrame(
    {
        "col1": [
            {"key1": "value1", "key2": "value2"},
            {"key1": "value3", "key2": "value4"},
        ],
        "col2": [{}, {}],
    }
)

d = pd.DataFrame(
    [
        {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "sparse_values": {"inices": [1, 2, 3], "values": [0.1, 0.2, 0.3]},
            "metadata": {"title": "title1", "url": "url1"},
            "blob": {"text": "text1"},
        },
        {
            "id": "2",
            "values": [0.4, 0.5, 0.6],
            "sparse_values": {"inices": [4, 5, 6], "values": [0.4, 0.5, 0.6]},
            "metadata": {"title": "title2", "url": "url2"},
            "blob": {"text": "text2"},
        },
    ]
)

q = pd.DataFrame(
    [
        {
            "vector": [0.1, 0.2, 0.3],
            "sparse_vector": {"inices": [1, 2, 3], "values": [0.1, 0.2, 0.3]},
            "filter": "filter1",
            "top_k": 1,
            "blob": None,
        },
        {
            "vector": [0.4, 0.5, 0.6],
            "sparse_vector": {"inices": [4, 5, 6], "values": [0.4, 0.5, 0.6]},
            "filter": "filter2",
            "top_k": 2,
            "blob": None,
        },
    ]
)



def test_single_key():
    result = transfer_keys_vectorized(df, "col1", "col2", ["key1"])
    expected = pd.DataFrame(
        {
            "col1": [{"key2": "value2"}, {"key2": "value4"}],
            "col2": [{"key1": "value1"}, {"key1": "value3"}],
        }
    )
    assert_frame_equal(result, expected)


def test_multiple_keys():
    result = transfer_keys_vectorized(df, "col1", "col2", ["key1", "key2"])
    expected = pd.DataFrame(
        {
            "col1": [{}, {}],
            "col2": [
                {"key1": "value1", "key2": "value2"},
                {"key1": "value3", "key2": "value4"},
            ],
        }
    )
    assert_frame_equal(result, expected)


def test_nonexistent_key():
    assert_frame_equal(df, transfer_keys_vectorized(df, "col1", "col2", ["nonexistent"]))


def test_datastes():
    metadata = DatasetMetadata(
        name="dataset_name",
        created_at="2021-01-01 00:00:00.000000",
        documents=2,
        queries=2,
        dense_model=DenseModelMetadata(
            name="ada2",
            dimension=2,
        ),
    )
    ds = Dataset.from_pandas(documents=d, queries=q, metadata=metadata)

    result = import_documents_keys_from_blob_to_metadata(ds, ["text"])

    d2 = pd.DataFrame(
        [
            {
                "id": "1",
                "values": [0.1, 0.2, 0.3],
                "sparse_values": {"inices": [1, 2, 3], "values": [0.1, 0.2, 0.3]},
                "metadata": {"title": "title1", "url": "url1", "text": "text1"},
                "blob": {},
            },
            {
                "id": "2",
                "values": [0.4, 0.5, 0.6],
                "sparse_values": {"inices": [4, 5, 6], "values": [0.4, 0.5, 0.6]},
                "metadata": {"title": "title2", "url": "url2", "text": "text2"},
                "blob": {},
            },
        ]
    )

    expected = Dataset.from_pandas(documents=d2, queries=q, metadata=metadata)

    assert_datasets_equal(result, expected)