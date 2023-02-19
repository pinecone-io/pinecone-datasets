from polars.datatypes import Utf8, Int64, Float32, List, Struct, Field


class Storage:
    base_path: str = "gs://pinecone-datasets-dev"


class Schema:
    documents = {
        "id": Utf8,
        "values": List(Float32),
        "sparse_values": Struct([Field('indices', List(Int64)), Field('values', List(Float32))])
    }
    queries = {
        "id": Utf8,
        "values": List(Float32),
        "sparse_values": Struct([Field('indices', List(Int64)), Field('values', List(Float32))]),
        "top_k": Int64
    }