from polars.datatypes import Utf8, Float32, List, Struct, Field, UInt32


class Storage:
    base_path: str = "gs://pinecone-datasets-dev"


class Schema:
    documents = {
        "id": Utf8,
        "values": List(Float32),
        "sparse_values": Struct(
            [Field("indices", List(UInt32)), Field("values", List(Float32))]
        ),
    }
    documents_select_columns = ["id", "values", "sparse_values", "metadata"]
    queries = {
        "values": List(Float32),
        "sparse_values": Struct(
            [Field("indices", List(UInt32)), Field("values", List(Float32))]
        ),
        "top_k": UInt32,
    }
    queries_select_columns = ["values", "sparse_values", "filter", "top_k"]
