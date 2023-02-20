import pandas as pd
import polars as pl
from pinecone_datasets import __version__, load_dataset, list_datasets, Dataset


def test_version():
    assert __version__ == '0.1.6-alpha'


def test_load_dataset_pandas():
    from pinecone_datasets import load_dataset
    ds = load_dataset("cc-news_msmarco-MiniLM-L6-cos-v5")
    assert ds.documents.shape[0] == 100000
    assert ds.documents.shape[1] == 5
    assert isinstance(ds.documents, pd.DataFrame)

def test_load_dataset_polars():
    from pinecone_datasets import load_dataset
    ds = load_dataset("cc-news_msmarco-MiniLM-L6-cos-v5", engine="polars")
    assert ds.documents.shape[0] == 100000
    assert ds.documents.shape[1] == 5
    assert isinstance(ds.documents, pl.DataFrame)

def test_list_datasets():
    from pinecone_datasets import list_datasets
    lst = list_datasets()
    assert len(lst) > 0
    assert isinstance(lst, list)
    assert isinstance(lst[0], str)
    assert "cc-news_msmarco-MiniLM-L6-cos-v5" in lst

def test_iter_documents_pandas(tmpdir):
    from pinecone_datasets import Dataset
    data = [
        {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "sparse_values": {"1": 0.1, "2": 0.2, "3": 0.3},
            "metadata": {"title": "title1", "url": "url1"},
        },
        {
            "id": "2",
            "values": [0.4, 0.5, 0.6],
            "sparse_values": {"4": 0.4, "5": 0.5, "6": 0.6},
            "metadata": {"title": "title2", "url": "url2"},
        }
    ]

    dataset_name = "test_dataset"
    dataset_path = tmpdir.mkdir(dataset_name)
    documents_path = dataset_path.mkdir("documents")
    pd.DataFrame(data).to_parquet(documents_path.join("part-0.parquet"))

    ds = Dataset(dataset_name, base_path=str(tmpdir))
    assert ds.documents.shape[0] == 2


def is_dicts_equal(d1, d2):
    return d1.keys() == d2.keys() and all(d1[k] == d2[k] for k in d1)

def download_json_from_https(url):
    import requests
    return requests.get(url).json()

def test_catalog():
    from pinecone_datasets import catalog
    catalog_as_dict = download_json_from_https("https://storage.googleapis.com/pinecone-datasets-dev/catalog.json")
    for dataset in catalog.list_datasets():
        assert dataset in [c["name"] for c in catalog_as_dict]