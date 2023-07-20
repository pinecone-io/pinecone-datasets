import os
import time
from importlib.metadata import version

if version("pinecone-client").startswith("3"):
    from pinecone import Client as PC, Index
elif version("pinecone-client").startswith("2"):
    import pinecone as PC

    try:
        from pinecone import GRPCIndex as Index
    except ImportError:
        from pinecone import Index
from pinecone_datasets import list_datasets, load_dataset

from tests.test_public_datasets import deep_list_cmp, deep_list_cmp_approx


class TestPinecone:
    def setup_method(self):
        if version("pinecone-client").startswith("3"):
            self.client = PC()
        elif version("pinecone-client").startswith("2"):
            PC.init()
            self.client = PC
        self.index_name = f"quora-index-{os.environ['CLIENT_VERSION']}-{os.environ['PY_VERSION'].replace('.', '-')}"
        self.dataset_size = 100000
        self.dataset_dim = 384
        self.tested_dataset = "quora_all-MiniLM-L6-bm25-100K"
        self.ds = load_dataset(self.tested_dataset)

    def test_large_dataset_upsert_to_pinecone_with_creating_index(self):
        print(f"Testing dataset {self.tested_dataset} with index {self.index_name}")

        if self.index_name in self.client.list_indexes():
            print(f"Deleting index {self.index_name}")
            self.client.delete_index(index_name)

            while self.index_name in self.client.list_indexes():
                print(f"Waiting for index {self.index_name} to be deleted")
                time.sleep(5)
        try:
            self.ds.to_pinecone_index(
                index_name=self.index_name, batch_size=300, concurrency=1
            )
            index = self.client.Index(self.index_name)

            assert self.index_name in self.client.list_indexes()
            assert self.client.describe_index(self.index_name).name == self.index_name
            assert (
                self.client.describe_index(self.index_name).dimension
                == self.dataset_dim
            )

            # Wait for index to be ready
            time.sleep(60)
            assert index.describe_index_stats().total_vector_count == self.dataset_size
            fetch_results_values = (
                index.fetch(ids=["1"])["1"].values
                if version("pinecone-client").startswith("3")
                else index.fetch(ids=["1"])["vectors"]["1"].values
            )
            assert deep_list_cmp_approx(
                fetch_results_values,
                self.ds.documents.loc[0].values[1].tolist(),
            )
        finally:
            if self.index_name in self.client.list_indexes():
                print(f"Deleting index {self.index_name}")
                self.client.delete_index(self.index_name)

    def test_dataset_upsert_to_existing_index(self):
        try:
            # create an index
            this_test_index = self.index_name + "-precreated"
            self.client.create_index(name=this_test_index, dimension=self.dataset_dim)
            # check that index exists
            assert this_test_index in self.client.list_indexes()

            # check that index is empty
            assert (
                self.client.Index(this_test_index)
                .describe_index_stats()
                .total_vector_count
                == 0
            )
            # upsert dataset to index
            self.ds.to_pinecone_index(
                index_name=this_test_index,
                batch_size=300,
                concurrency=1,
                should_create=False,
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
                concurrency=1,
                should_create=False,
                namespace=namespace,
            )

            # Wait for index to be ready
            time.sleep(60)
            assert (
                index.describe_index_stats().total_vector_count == self.dataset_size * 2
            )
            assert (
                index.describe_index_stats().namespaces[namespace].vector_count
                == self.dataset_size
            )

        finally:
            if this_test_index in self.client.list_indexes():
                print(f"Deleting index {this_test_index}")
                self.client.delete_index(this_test_index)
