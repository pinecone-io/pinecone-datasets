import pandas as pd
from pandas.testing import assert_frame_equal as pd_assert_frame_equal

from pinecone_datasets import Dataset
from pinecone_datasets.catalog import DatasetMetadata, DenseModelMetadata

d = pd.DataFrame(
    [
        {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "sparse_values": {"inices": [1, 2, 3], "values": [0.1, 0.2, 0.3]},
            "metadata": {"title": "title1", "url": "url1"},
            "blob": None,
        },
        {
            "id": "2",
            "values": [0.4, 0.5, 0.6],
            "sparse_values": {"inices": [4, 5, 6], "values": [0.4, 0.5, 0.6]},
            "metadata": {"title": "title2", "url": "url2"},
            "blob": None,
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


def test_io_cloud_storage():
    dataset_name = "test_io_dataset"
    dataset_path = f"s3://ram-datasets/{dataset_name}/"
    metadata = DatasetMetadata(
        name=dataset_name,
        created_at="2021-01-01 00:00:00.000000",
        documents=2,
        queries=2,
        dense_model=DenseModelMetadata(
            name="ada2",
            dimension=2,
        ),
    )
    ds = Dataset.from_pandas(documents=d, queries=q, metadata=metadata)
    ds.save_to_path(str(dataset_path), endpoint_url="https://storage.googleapis.com")
    loaded_ds = Dataset.from_path(
        str(dataset_path), endpoint_url="https://storage.googleapis.com"
    )
    assert loaded_ds.metadata == metadata
    pd_assert_frame_equal(loaded_ds.documents, ds.documents)
    pd_assert_frame_equal(loaded_ds.queries, ds.queries)

    loaded_ds._fs.rm(dataset_path, recursive=True)
