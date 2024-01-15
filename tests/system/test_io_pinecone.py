import os
import time
import uuid

import pandas as pd
import pytest

import pinecone as pc
from pinecone import ServerlessSpec, PodSpec
from pinecone_datasets import (
    load_dataset,
    DatasetMetadata,
    Dataset,
    DenseModelMetadata,
)

from typing import List

from tests.system.test_public_datasets import approx_deep_list_cmp


@pytest.fixture
def spec_type(request):
    return request.param

class TestPinecone:
    def setup_method(self):
        # Prep Pinecone Dataset and Index for testing
        self.client = pc.Pinecone()
        self.index_name = f"quora-index-{os.environ['PY_VERSION'].replace('.', '-')}-{uuid.uuid4().hex[-6:]}"
        self.dataset_size = 100000
        self.dataset_dim = 384
        self.tested_dataset = "quora_all-MiniLM-L6-bm25-100K"
        self.ds = load_dataset(self.tested_dataset)

        # Prep Local Dataasets with different metadata combinations

        data_documents = [
            {
                "id": "1",
                "values": [0.1, 0.2, 0.3],
                "sparse_values": {"indices": [1, 2, 3], "values": [0.1, 0.2, 0.3]},
                "metadata": {"url": "url1"},
                "blob": None,
            },
            {
                "id": "2",
                "values": [0.4, 0.5, 0.6],
                "sparse_values": {"indices": [4, 5, 6], "values": [0.4, 0.5, 0.6]},
                "metadata": {"title": "title2"},
                "blob": None,
            },
            {
                "id": "3",
                "values": [0.7, 0.8, 0.9],
                "sparse_values": {"indices": [7, 8, 9], "values": [0.7, 0.8, 0.9]},
                "metadata": {},
                "blob": None,
            },
        ]

        data_queries = [
            {
                "vector": [0.11, 0.21, 0.31],
                "filter": {"url": {"$eq": "url1"}},
                "top_k": 1,
            },
            {
                "vector": [0.41, 0.51, 0.61],
                "sparse_vector": {"indices": [4, 6], "values": [0.4, 0.6]},
                "metadata": {"title": {"$eq": "title2"}, "url": {"$neq": "url2"}},
                "top_k": 2,
            },
        ]

        data_metadata = DatasetMetadata(
            name="test",
            documents=3,
            queries=2,
            dense_model=DenseModelMetadata(name="test", dimension=3),
            description="test",
            tags=["test"],
            source="test",
            license="test",
            authors=["test"],
            links={"test": "test"},
            version="test",
            created_at="test",
            updated="test",
            schema={"test": "test"},
        )

        self.ds_local = Dataset.from_pandas(
            documents=pd.DataFrame(data_documents),
            queries=pd.DataFrame(data_queries),
            metadata=data_metadata,
        )

        self.index_name_local = f"test-index-{os.environ['PY_VERSION'].replace('.', '-')}-{uuid.uuid4().hex[:6]}"

    def test_local_dataset_with_metadata(self, tmpdir):
        print(
            f"Testing dataset {self.tested_dataset} with index {self.index_name_local}"
        )

        self.ds_local.to_pinecone_index(index_name=self.index_name_local, batch_size=3)
        index = self.client.Index(self.index_name_local)

        assert self.index_name_local in self._get_index_list()
        assert (
            self.client.describe_index(self.index_name_local).name
            == self.index_name_local
        )
        assert (
            self.client.describe_index(self.index_name_local).dimension
            == self.ds_local.metadata.dense_model.dimension
        )

        # Wait for index to be ready
        time.sleep(60)
        assert (
            index.describe_index_stats().total_vector_count
            == self.ds_local.metadata.documents
        )

        dataset_name = "test_local_dataset_with_metadata"
        dataset_path = tmpdir.mkdir(dataset_name)

        self.ds_local.to_path(str(dataset_path))

        loaded_ds = Dataset.from_path(str(dataset_path))
        assert loaded_ds.metadata == self.ds_local.metadata

        pd.testing.assert_frame_equal(loaded_ds.documents, self.ds_local.documents)

        pd.testing.assert_frame_equal(loaded_ds.queries, self.ds_local.queries)

    def test_large_dataset_upsert_to_pinecone_with_creating_index(self):
        print(f"Testing dataset {self.tested_dataset} with index {self.index_name}")

        self.ds.to_pinecone_index(index_name=self.index_name, batch_size=300)
        index = self.client.Index(self.index_name)

        assert self.index_name in self._get_index_list()
        assert self.client.describe_index(self.index_name).name == self.index_name
        assert self.client.describe_index(self.index_name).dimension == self.dataset_dim

        # Wait for index to be ready
        time.sleep(60)
        assert index.describe_index_stats().total_vector_count == self.dataset_size

        assert approx_deep_list_cmp(
            index.fetch(ids=["1"])["vectors"]["1"].values,
            self.ds.documents.loc[0].values[1].tolist(),
        )

    @pytest.mark.parametrize("spec_type", ["pod", "serverless"], indirect=True)
    def test_dataset_upsert_to_existing_index(self, spec_type):
        # create an index
        this_test_index = self.index_name + "-precreated"  
        if spec_type == "serverless":
            spec = ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-west-2"),
            )
        elif spec_type == "pod":
            spec = PodSpec(environment=os.environ["PINECONE_ENVIRONMENT"])
        else:
            raise ValueError(f"Unknown spec type {spec_type}")
        self.client.create_index(
            name=this_test_index,
            dimension=self.dataset_dim,
            spec=spec
        )
        print(f"Created v3 index {this_test_index} with spec {spec}")

        # check that index exists
        assert this_test_index in self._get_index_list()

        # check that index is empty
        assert (
            self.client.Index(this_test_index).describe_index_stats().total_vector_count
            == 0
        )
        # upsert dataset to index
        self.ds.to_pinecone_index(
            index_name=this_test_index,
            batch_size=300,
            should_create_index=False,
        )
        index = self.client.Index(this_test_index)

        # Wait for index to be ready
        time.sleep(60)
        assert index.describe_index_stats().total_vector_count == self.dataset_size
        assert (
            index.describe_index_stats().namespaces[""].vector_count
            == self.dataset_size
        )

        # upsert dataset to index at a specific namespace
        namespace = "test"
        self.ds.to_pinecone_index(
            index_name=this_test_index,
            batch_size=300,
            should_create_index=False,
            namespace=namespace,
        )

        # Wait for index to be ready
        time.sleep(60)
        assert index.describe_index_stats().total_vector_count == self.dataset_size * 2
        assert (
            index.describe_index_stats().namespaces[namespace].vector_count
            == self.dataset_size
        )
    
    def _get_index_list(self) -> List[str]:
        return [i["name"] for i in self.client.list_indexes()]

    def teardown_method(self):
        if self.index_name in self._get_index_list():
            print(f"Deleting index {self.index_name}")
            self.client.delete_index(self.index_name)

        if self.index_name + "-precreated" in self._get_index_list():
            print(f"Deleting index {self.index_name}-precreated")
            self.client.delete_index(self.index_name + "-precreated")

        if self.index_name_local in self._get_index_list():
            print(f"Deleting index {self.index_name_local}")
            self.client.delete_index(self.index_name_local)
