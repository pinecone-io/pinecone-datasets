from polars.datatypes import DataType, Utf8, Int32, Float32, List, Struct, Field

class Storage:
    base_path: str = "gs://pinecone-datasets-dev"


class Schema:
    documents = {
        "id": Utf8,
        "values": List(Float32),
        "sparse_values": Struct([Field('indices', List(Int32)), Field('values', List(Float32))])
    }
    queries = {
        "id": Utf8,
        "values": List(Float32),
        "sparse_values": Struct([Field('indices', List(Int32)), Field('values', List(Float32))])
    }