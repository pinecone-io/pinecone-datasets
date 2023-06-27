# from polars.datatypes import Utf8, Float32, List, Struct, Field, UInt32


class Storage:
    endpoint: str = "gs://pinecone-datasets-dev"


class Schema:
    class Names:
        documents = [
            ("id", False, None),
            ("values", False, None),
            ("sparse_values", True, None),
            ("metadata", True, None),
            ("blob", True, None),
        ]
        queries = [
            ("vector", False, None),
            ("sparse_vector", True, None),
            ("filter", True, None),
            ("top_k", False, 5),
            ("blob", True, None),
        ]

    # documents = {
    #     "id": Utf8,
    #     "values": List(Float32),
    #     "sparse_values": Struct(
    #         [Field("indices", List(UInt32)), Field("values", List(Float32))]
    #     ),
    # }
    documents_select_columns = ["id", "values", "sparse_values", "metadata"]

    # queries = {
    #     "vector": List(Float32),
    #     "sparse_vector": Struct(
    #         [Field("indices", List(UInt32)), Field("values", List(Float32))]
    #     ),
    #     "top_k": UInt32,
    # }
    queries_select_columns = ["vector", "sparse_vector", "filter", "top_k"]
