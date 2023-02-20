from dataclasses import dataclass
import json
from typing import List, Optional

import gcsfs
import polars as pl
from google.cloud import storage
from pydantic import BaseModel, Field

class DenseModelMetadata(BaseModel):
    name: str
    tokenizer: Optional[str]
    dimension: int

class SparseModelMetdata(BaseModel):
    name: str
    tokenizer: Optional[str]

class DatasetMetadata(BaseModel):
    name: str
    created_at: str
    dense_model: DenseModelMetadata
    sparse_model: SparseModelMetdata

class Catalog(BaseModel):
    datasets: list[DatasetMetadata]

    def load() -> "Catalog":
        gcs_file_system = gcsfs.GCSFileSystem(token='anon')
        gcs_json_path = "gs://pinecone-datasets-dev/catalog.json"
        with gcs_file_system.open(gcs_json_path) as f:
            _catalog = pl.from_dicts(json.load(f))
        return Catalog.parse_obj({"datasets": _catalog.to_dicts()})

    def list_datasets(self) -> list:
        return [dataset.name for dataset in self.datasets]


# class Catalog(object):
#     def __init__(self) -> None:
#         gcs_file_system = gcsfs.GCSFileSystem(token='anon')
#         gcs_json_path = "gs://pinecone-datasets-dev/catalog.json"
#         with gcs_file_system.open(gcs_json_path) as f:
#             self._catalog = pl.from_dicts(json.load(f))

#     def is_in_catalog(self, dataset_id: str) -> bool:
#         filtered_catalog = self._catalog.filter(pl.col("name") == dataset_id)
#         if filtered_catalog.shape[0] == 0:
#             return False
#         elif filtered_catalog.shape[0] == 1:
#             return True
#         else:
#             raise ValueError("There is more than one dataset with the same name")
        
#     def get_dataset(self, dataset_id: str) -> pl.DataFrame:
#         if self.is_in_catalog(dataset_id):
#             return self._catalog.filter(pl.col("name") == dataset_id).to_dict()
#         else:
#             raise ValueError("Dataset not found in catalog")

#     def list_datasets(self) -> list:
#         return self._catalog["name"].to_list()