import os
import time

from pinecone import Client
from pinecone_datasets import list_datasets, load_dataset

from tests.test_public_datasets import deep_list_cmp


def test_large_dataset_upsert_to_pinecone():
    tested_dataset = "quora_all-MiniLM-L6-bm25-100K"
    index_name = f"quora-index-{os.environ['PY_VERSION'].replace('.', '-')}"
    dataset_size = 100000
    dataset_dim = 384

    print(f"Testing dataset {tested_dataset} with index {index_name}")
    ds = load_dataset(tested_dataset)

    client = Client()

    assert ds.documents.shape[0] == dataset_size

    if index_name in client.list_indexes():
        print(f"Deleting index {index_name}")
        client.delete_index(index_name)

        while index_name in client.list_indexes():
            print(f"Waiting for index {index_name} to be deleted")
            time.sleep(5)
    try:
        ds.to_pinecone_index(index_name=index_name, batch_size=300, concurrency=1)
        index = client.get_index(index_name)

        assert index_name in client.list_indexes()
        assert client.describe_index(index_name).name == index_name
        assert client.describe_index(index_name).dimension == dataset_dim

        # Wait for index to be ready
        time.sleep(60)
        assert index.describe_index_stats().total_vector_count == dataset_size

        assert deep_list_cmp(
            index.fetch(ids=["1"])["1"].values, ds.documents.loc[0].values[1].tolist()
        )
    finally:
        if index_name in client.list_indexes():
            print(f"Deleting index {index_name}")
            client.delete_index(index_name)
