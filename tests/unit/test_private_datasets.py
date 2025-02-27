import pytest
import os

from pinecone_datasets import list_datasets, load_dataset, Dataset
from tests.system.test_public_datasets import deep_list_cmp


@pytest.mark.skip(reason="Need to figure out credentials to run these tests")
class TestPrivateDatasets:
    def test_list_private_datasets(self):
        os.environ["DATASETS_CATALOG_BASEPATH"] = "s3://ram-datasets"
        lst = list_datasets(endpoint_url="https://storage.googleapis.com")
        print(lst)
        del os.environ["DATASETS_CATALOG_BASEPATH"]
        assert "test_dataset" in lst

    def test_load_private_dataset(self):
        os.environ["DATASETS_CATALOG_BASEPATH"] = "s3://ram-datasets"
        ds = load_dataset("test_dataset", endpoint_url="https://storage.googleapis.com")
        assert isinstance(ds, Dataset)
        assert ds.queries.shape[0] == 2
        assert ds.documents.shape[0] == 2
        assert deep_list_cmp(
            ds.documents.columns, ["id", "values", "sparse_values", "metadata"]
        )
        del os.environ["DATASETS_CATALOG_BASEPATH"]

    def test_dataset_from_path(self):
        dataset_path = "s3://ram-datasets/test_dataset"
        ds = Dataset.from_path(
            dataset_path, endpoint_url="https://storage.googleapis.com"
        )
        assert isinstance(ds, Dataset)
        assert ds.queries.shape[0] == 2
        assert ds.documents.shape[0] == 2
        assert deep_list_cmp(
            ds.documents.columns, ["id", "values", "sparse_values", "metadata"]
        )
