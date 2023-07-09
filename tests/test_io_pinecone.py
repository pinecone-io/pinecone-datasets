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

from tests.test_public_datasets import deep_list_cmp


class TestPinecone:
    
    def setup_method(self):
        if version("pinecone-client").startswith("3"):
            self.client = PC()
        elif version("pinecone-client").startswith("2"):
            PC.init()
            self.client = PC
        self.index_name = f"quora-index-{os.environ['PY_VERSION'].replace('.', '-')}"
        self.dataset_size = 100000
        self.dataset_dim = 384
        self.tested_dataset = "quora_all-MiniLM-L6-bm25-100K"
        
        # create an index
        self.client.create_index(name=self.index_name + "-precreated", dimension=self.dataset_dim)
        self.ds = load_dataset(self.tested_dataset)
        assert self.ds.documents.shape[0] == self.dataset_size
        

    def test_large_dataset_upsert_to_pinecone_with_creating_index(self):
        print(f"Testing dataset {self.tested_dataset} with index {self.index_name}")
        

        if self.index_name in self.client.list_indexes():
            print(f"Deleting index {self.index_name}")
            self.client.delete_index(index_name)

            while self.index_name in self.client.list_indexes():
                print(f"Waiting for index {self.index_name} to be deleted")
                time.sleep(5)
        try:
            self.ds.to_pinecone_index(index_name=self.index_name, batch_size=300, concurrency=1)
            index = self.client.Index(self.index_name)

            assert self.index_name in self.client.list_indexes()
            assert self.client.describe_index(self.index_name).name == self.index_name
            assert self.client.describe_index(self.index_name).dimension == self.dataset_dim

            # Wait for index to be ready
            time.sleep(60)
            assert index.describe_index_stats().total_vector_count == self.dataset_size

            assert deep_list_cmp(
                index.fetch(ids=["1"])["1"].values, selfds.documents.loc[0].values[1].tolist()
            )
        finally:
            if self.index_name in self.client.list_indexes():
                print(f"Deleting index {self.index_name}")
                self.client.delete_index(self.index_name)


    def test_dataset_upsert_to_existing_index(self):
        this_test_index = self.index_name + "-precreated"

        # check that index exists
        assert this_test_index in self.client.list_indexes()

        # check that index is empty
        assert self.client.Index(this_test_index).describe_index_stats().total_vector_count == 0

        # upsert dataset to index
        self.ds.to_pinecone_index(index_name=this_test_index, batch_size=300, concurrency=1, create_index=False)
        index = self.client.Index(this_test_index)

        # Wait for index to be ready
        time.sleep(60)
        assert index.describe_index_stats().total_vector_count == self.dataset_size
        assert index.describe_index_stats().namespaces[''].vector_count == self.dataset_size

        # upsert dataset to index at a specific namespace
        namespace = "test"
        self.ds.to_pinecone_index(index_name=this_test_index, batch_size=300, concurrency=1, create_index=False, namespace=namespace)
        
        # Wait for index to be ready
        time.sleep(60)
        assert index.describe_index_stats().total_vector_count == self.dataset_size * 2
        assert index.describe_index_stats().namespaces[namespace].vector_count == self.dataset_size


    def teardown_method(self):

        def delete_one(index_name):
            if index_name in self.client.list_indexes():
                print(f"Deleting index {index_name}")
                self.client.delete_index(index_name)

                while index_name in self.client.list_indexes():
                    print(f"Waiting for index {index_name} to be deleted")
                    time.sleep(5)
        
        for index_name in [self.index_name, self.index_name + "-precreated"]:
            delete_one(index_name)
        
