import json
import pytest

import pandas as pd
from pydantic import ValidationError

from pinecone_datasets import Dataset, DatasetMetadata, DenseModelMetadata


def test_datasets_schema_name_happy(tmpdir):
    documents_data = [
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

    dataset_name = "test_dataset"
    dataset_path = tmpdir.mkdir(dataset_name)
    documents_path = dataset_path.mkdir("documents")
    pd.DataFrame(documents_data).to_parquet(documents_path.join("part-0.parquet"))

    queries_data = [
        {
            "vector": [0.1, 0.2, 0.3],
            "sparse_vector": {"indices": [1, 2, 3], "values": [0.1, 0.2, 0.3]},
            "filter": {"filter1": {"$eq": "filter1"}},
            "top_k": 1,
            "blob": None,
        },
        {
            "vector": [0.4, 0.5, 0.6],
            "sparse_vector": {"indices": [4, 5, 6], "values": [0.4, 0.5, 0.6]},
            "filter": {"filter2": {"$eq": "filter2"}},
            "top_k": 2,
            "blob": None,
        },
    ]

    queries_path = dataset_path.mkdir("queries")
    pd.DataFrame(queries_data).to_parquet(queries_path.join("part-0.parquet"))

    metadata: DatasetMetadata = DatasetMetadata(
        name=dataset_name,
        created_at="2021-01-01 00:00:00.000000",
        documents=2,
        queries=2,
        dense_model=DenseModelMetadata(
            name="ada2",
            dimension=2,
        ),
    )

    with open(dataset_path.join("metadata.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    ds = Dataset.from_path(str(dataset_path))
    assert isinstance(ds, Dataset)
    assert ds.queries.shape[0] == 2
    assert ds.documents.shape[0] == 2


def test_datasets_schema_name_documents_missing_propery(tmpdir):
    documents_data = [
        {
            "id": "1",
            "sparse_values": {"indices": [1, 2, 3], "values": [0.1, 0.2, 0.3]},
            "metadata": {"title": "title1", "url": "url1"},
            "blob": None,
        },
        {
            "id": "2",
            "sparse_values": {"indices": [4, 5, 6], "values": [0.4, 0.5, 0.6]},
            "metadata": {"title": "title2", "url": "url2"},
            "blob": None,
        },
    ]

    dataset_name = "test_dataset"
    dataset_path = tmpdir.mkdir(dataset_name)
    documents_path = dataset_path.mkdir("documents")
    pd.DataFrame(documents_data).to_parquet(documents_path.join("part-0.parquet"))

    queries_data = [
        {
            "vector": [0.1, 0.2, 0.3],
            "sparse_vector": {"indices": [1, 2, 3], "values": [0.1, 0.2, 0.3]},
            "filter": {"filter1": {"$eq": "filter1"}},
            "top_k": 1,
            "blob": None,
        },
        {
            "vector": [0.4, 0.5, 0.6],
            "sparse_vector": {"indices": [4, 5, 6], "values": [0.4, 0.5, 0.6]},
            "filter": {"filter2": {"$eq": "filter2"}},
            "top_k": 2,
            "blob": None,
        },
    ]

    queries_path = dataset_path.mkdir("queries")
    pd.DataFrame(queries_data).to_parquet(queries_path.join("part-0.parquet"))

    metadata: DatasetMetadata = DatasetMetadata(
        name=dataset_name,
        created_at="2021-01-01 00:00:00.000000",
        documents=2,
        queries=2,
        dense_model=DenseModelMetadata(
            name="ada2",
            dimension=2,
        ),
    )

    with open(dataset_path.join("metadata.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    with pytest.raises(ValueError):
        ds = Dataset.from_path(str(dataset_path))
        assert isinstance(ds, Dataset)
        assert ds.queries.shape[0] == 2
        assert ds.documents.shape[0] == 2


def test_datasets_schema_name_queries_missing_propery(tmpdir):
    documents_data = [
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

    dataset_name = "test_dataset"
    dataset_path = tmpdir.mkdir(dataset_name)
    documents_path = dataset_path.mkdir("documents")
    pd.DataFrame(documents_data).to_parquet(documents_path.join("part-0.parquet"))

    queries_data = [
        {
            "sparse_vector": {"indices": [1, 2, 3], "values": [0.1, 0.2, 0.3]},
            "filter": {"filter1": {"$eq": "filter1"}},
            "top_k": 1,
        },
        {
            "sparse_vector": {"indices": [4, 5, 6], "values": [0.4, 0.5, 0.6]},
            "filter": {"filter2": {"$eq": "filter2"}},
            "top_k": 2,
        },
    ]

    queries_path = dataset_path.mkdir("queries")
    pd.DataFrame(queries_data).to_parquet(queries_path.join("part-0.parquet"))

    metadata: DatasetMetadata = DatasetMetadata(
        name=dataset_name,
        created_at="2021-01-01 00:00:00.000000",
        documents=2,
        queries=2,
        dense_model=DenseModelMetadata(
            name="ada2",
            dimension=2,
        ),
    )

    with open(dataset_path.join("metadata.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    with pytest.raises(ValueError):
        ds = Dataset.from_path(str(dataset_path))
        assert isinstance(ds, Dataset)
        assert ds.queries.shape[0] == 2
        assert ds.documents.shape[0] == 2


def test_datasets_schema_metadata_wrong(tmpdir):
    with pytest.raises(ValidationError):
        metadata: DatasetMetadata = DatasetMetadata(
            created_at="2021-01-01 00:00:00.000000",
            documents=2,
            queries=2,
            dense_model=DenseModelMetadata(
                name="ada2",
                dimension=2,
            ),
        )
