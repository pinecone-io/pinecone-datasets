import os

import pandas as pd
import polars as pl
import numpy as np
from polars.testing import assert_frame_equal as pl_assert_frame_equal
from pandas.testing import assert_frame_equal as pd_assert_frame_equal
import pytest
from pinecone_datasets import __version__, load_dataset, list_datasets, Dataset

WARN_MESSAGE = "Pinecone Datasets is a new and experimental library. The API is subject to change without notice."

test_base_path = "gs://ram-datasets"
test_dataset = "quora_all-MiniLM-L6-bm25"


def test_version():
    assert __version__ == "0.3.0-alpha"


def test_load_dataset_pandas():
    lst = list_datasets()
    assert test_dataset in lst
    ds = load_dataset(test_dataset)
    assert ds.documents.shape[0] == 522931 and len(ds) == 522931
    assert ds.documents.shape[0] == ds._metadata.documents
    assert ds.documents.shape[1] == 5
    assert isinstance(ds.documents, pd.DataFrame)
    assert isinstance(ds.head(), pd.DataFrame)
    assert ds.head().shape[0] == 5
    assert ds.head().shape[1] == 5
    pd_assert_frame_equal(ds["documents"], ds.documents)
    pd_assert_frame_equal(ds.head(), ds.documents.head())

    assert ds._metadata.name == test_dataset
    assert ds._metadata.queries == 15000


def test_load_dataset_polars():
    ds = Dataset(test_dataset, engine="polars")
    assert ds.documents.shape[0] == 522931
    assert ds.documents.shape[1] == 5
    assert isinstance(ds.documents, pl.DataFrame)
    assert isinstance(ds.head(), pl.DataFrame)
    assert ds.head().shape[0] == 5
    assert ds.head().shape[1] == 5
    pl_assert_frame_equal(ds.head(), ds.documents.head())


def test_list_datasets():
    lst = list_datasets()
    assert len(lst) > 0
    assert isinstance(lst, list)
    assert isinstance(lst[0], str)
    assert test_dataset in lst


def test_load_dataset_does_not_exists():
    with pytest.raises(FileNotFoundError):
        ds = load_dataset("does_not_exists")
    with pytest.raises(FileNotFoundError):
        ds = Dataset("does_not_exists")


def test_iter_documents_pandas(tmpdir):
    data = [
        {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "sparse_values": {"inices": [1, 2, 3], "values": [0.1, 0.2, 0.3]},
            "metadata": {"title": "title1", "url": "url1"},
        },
        {
            "id": "2",
            "values": [0.4, 0.5, 0.6],
            "sparse_values": {"inices": [4, 5, 6], "values": [0.4, 0.5, 0.6]},
            "metadata": {"title": "title2", "url": "url2"},
        },
    ]

    dataset_name = "test_dataset"
    dataset_path = tmpdir.mkdir(dataset_name)
    documents_path = dataset_path.mkdir("documents")
    pd.DataFrame(data).to_parquet(documents_path.join("part-0.parquet"))

    ds = Dataset(dataset_name, endpoint=str(tmpdir))

    for i, d in enumerate(ds.iter_documents()):
        assert isinstance(d, list)
        assert len(d) == 1
        assert isinstance(d[0], dict)
        assert is_dicts_equal(d[0], data[i])
        break

    for d in ds.iter_documents(batch_size=2):
        assert is_dicts_equal(d[0], data[0])
        assert is_dicts_equal(d[1], data[1])
        break

    assert ds.documents.shape[0] == 2


def test_iter_queries_pandas(tmpdir):
    data = [
        {
            "vector": [0.1, 0.2, 0.3],
            "sparse_vector": {"inices": [1, 2, 3], "values": [0.1, 0.2, 0.3]},
            "filter": "filter1",
            "top_k": 1,
        },
        {
            "vector": [0.4, 0.5, 0.6],
            "sparse_vector": {"inices": [4, 5, 6], "values": [0.4, 0.5, 0.6]},
            "filter": "filter2",
            "top_k": 2,
        },
    ]

    dataset_name = "test_dataset"
    dataset_path = tmpdir.mkdir(dataset_name)
    queries_path = dataset_path.mkdir("queries")
    pd.DataFrame(data).to_parquet(queries_path.join("part-0.parquet"))

    ds = Dataset(dataset_name, endpoint=str(tmpdir))

    for i, d in enumerate(ds.iter_queries()):
        print(d)
        print(data[i])
        assert isinstance(d, dict)
        assert is_dicts_equal(d, data[i])

    assert ds.queries.shape[0] == 2


def is_dicts_equal(d1, d2):
    return d1.keys() == d2.keys() and recursive_dict_compare(d1, d2)


def deep_list_cmp(l1, l2):
    same = True
    for l, r in zip(l1, l2):
        same = same and l == r
    return same


def recursive_dict_compare(d1, d2):
    for k, v in d1.items():
        if isinstance(v, dict):
            return recursive_dict_compare(v, d2[k])
        elif isinstance(v, (list, np.ndarray)):
            return deep_list_cmp(v, d2[k])
        return v == d2[k]


def download_json_from_https(url):
    import requests

    return requests.get(url).json()


def test_catalog():
    from pinecone_datasets.catalog import Catalog

    catalog = Catalog.load()

    catalog_as_dict = download_json_from_https(
        "https://storage.googleapis.com/pinecone-datasets-dev/quora_all-MiniLM-L6-bm25/metadata.json"
    )
    found = False
    for dataset in catalog.list_datasets(as_df=False):
        if catalog_as_dict["name"] == dataset:
            found = True
            break
    assert found
