from datetime import datetime

import pandas as pd
from pandas.testing import assert_frame_equal as pd_assert_frame_equal

from pinecone_datasets import Dataset, list_datasets
from pinecone_datasets.catalog import DatasetMetadata, DenseModelMetadata
import os

GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not GOOGLE_APPLICATION_CREDENTIALS:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is not set")

d = pd.DataFrame(
    [
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
)

q = pd.DataFrame(
    [
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
)


class TestSaveDatasetToGCS:
    def test_io_cloud_storage_path(self):
        dataset_name = "test_io_dataset"
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
        dataset_path = f"gs://pinecone-datasets-test/unittests/{dataset_name}/{datetime.now().strftime('%Y%m%d%H%M%S')}"

        ds.to_path(
            dataset_path,
            endpoint_url="https://storage.googleapis.com",
            token=GOOGLE_APPLICATION_CREDENTIALS,
        )

        loaded_ds = Dataset.from_path(
            dataset_path,
            endpoint_url="https://storage.googleapis.com",
            token=GOOGLE_APPLICATION_CREDENTIALS,
        )
        assert loaded_ds.metadata == metadata
        pd_assert_frame_equal(loaded_ds.documents, ds.documents)
        pd_assert_frame_equal(loaded_ds.queries, ds.queries)

    def test_io_cloud_storage_catalog(self):
        dataset_name = "test_io_dataset"
        dataset_id = dataset_name + "_" + datetime.now().strftime("%Y%m%d%H%M%S")
        catalog_base_path = f"gs://pinecone-datasets-test/unittests/catalog/"
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
        ds.to_catalog(
            catalog_base_path=catalog_base_path,
            dataset_id=dataset_id,
            endpoint_url="https://storage.googleapis.com",
            token=GOOGLE_APPLICATION_CREDENTIALS,
        )

        # Check that the dataset is in the catalog
        os.environ["DATASETS_CATALOG_BASEPATH"] = catalog_base_path
        catalog = list_datasets(as_df=True, token=GOOGLE_APPLICATION_CREDENTIALS)
        assert dataset_id in catalog["name"].values

        loaded_ds = Dataset.from_catalog(
            dataset_id=dataset_id,
            catalog_base_path=catalog_base_path,
            endpoint_url="https://storage.googleapis.com",
            token=GOOGLE_APPLICATION_CREDENTIALS,
        )
        assert loaded_ds.metadata == metadata
        pd_assert_frame_equal(loaded_ds.documents, ds.documents)
        pd_assert_frame_equal(loaded_ds.queries, ds.queries)
