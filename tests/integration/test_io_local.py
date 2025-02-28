import pytest
import pandas as pd
import logging
from pandas.testing import assert_frame_equal as pd_assert_frame_equal

from pinecone_datasets import Dataset, Catalog, DenseModelMetadata, DatasetMetadata

logger = logging.getLogger(__name__)

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


class TestLocalIO:
    def test_io_write_to_local(self, tmpdir):
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
        assert ds._fs is None
        assert ds._dataset_path is None

        catalog_path = tmpdir.mkdir("catalog")
        catalog = Catalog(base_path=str(catalog_path))
        catalog.save_dataset(ds)

        loaded_ds = catalog.load_dataset(dataset_name)
        assert loaded_ds.metadata == metadata
        pd_assert_frame_equal(loaded_ds.documents, ds.documents)
        pd_assert_frame_equal(loaded_ds.queries, ds.queries)
        assert loaded_ds._fs is not None
        assert loaded_ds._dataset_path is not None

    def test_io_no_queries(self, tmpdir):
        dataset_name = "test_io_dataset_no_q"
        metadata = DatasetMetadata(
            name=dataset_name,
            created_at="2021-01-01 00:00:00.000000",
            documents=2,
            queries=0,
            dense_model=DenseModelMetadata(
                name="ada2",
                dimension=2,
            ),
        )
        ds = Dataset.from_pandas(documents=d, queries=None, metadata=metadata)
        assert ds._fs is None
        assert ds._dataset_path is None

        catalog_path = tmpdir.mkdir("catalog")
        catalog = Catalog(base_path=str(catalog_path))
        catalog.save_dataset(ds)

        loaded_ds = catalog.load_dataset(dataset_name)
        assert loaded_ds.metadata == metadata
        pd_assert_frame_equal(loaded_ds.documents, ds.documents)
        assert loaded_ds.queries.empty
        assert loaded_ds._fs is not None
        assert loaded_ds._dataset_path is not None

    def test_load_from_cloud_and_save_to_local(self, tmpdir):
        public_catalog = Catalog()
        ds = public_catalog.load_dataset("langchain-python-docs-text-embedding-ada-002")

        local_catalog_path = tmpdir.mkdir("catalog")
        local_catalog = Catalog(base_path=str(local_catalog_path))
        local_catalog.save_dataset(ds)

        logger.debug(f"wrote data to local_catalog_path: {str(local_catalog_path)}")

        loaded_ds = local_catalog.load_dataset(ds.metadata.name)
        # Assert frames have the same number of rows
        assert loaded_ds.documents.shape[0] == ds.documents.shape[0]
        assert loaded_ds.queries.shape[0] == ds.queries.shape[0]
        # Assert frames have the same columns
        assert loaded_ds.documents.columns.tolist() == ds.documents.columns.tolist()
        assert loaded_ds.queries.columns.tolist() == ds.queries.columns.tolist()
