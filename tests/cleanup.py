from pinecone import Client


client = Client()


def run():
    for index in client.list_indexes():
        print(f"Deleting index {index}")
        client.delete_index(index)
        print(f"Deleted index {index}")
