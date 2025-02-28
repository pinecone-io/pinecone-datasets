from datetime import datetime

import pandas as pd
import random
from pandas.testing import assert_frame_equal as pd_assert_frame_equal

from pinecone_datasets import (
    Dataset,
    Catalog,
    list_datasets,
    DatasetMetadata,
    DenseModelMetadata,
)
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
    def test_io_cloud_storage(self):
        dataset_name = "test_io_dataset_" + str(random.randint(0, 1000000))
        metadata = DatasetMetadata(
            name=dataset_name,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            documents=2,
            queries=2,
            dense_model=DenseModelMetadata(
                name="ada2",
                dimension=2,
            ),
        )
        ds = Dataset.from_pandas(documents=d, queries=q, metadata=metadata)

        catalog = Catalog(base_path="gs://pinecone-datasets-test/catalog")
        catalog.save_dataset(dataset=ds)

        loaded_ds = catalog.load_dataset(dataset_name)
        print(catalog.list_datasets(as_df=True))

        assert loaded_ds.metadata == metadata
        pd_assert_frame_equal(loaded_ds.documents, ds.documents)
        pd_assert_frame_equal(loaded_ds.queries, ds.queries)
