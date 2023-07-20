from pinecone import Client


client = Client()


def run():
    for index in client.list_indexes():
        client.delete_index(index=index)