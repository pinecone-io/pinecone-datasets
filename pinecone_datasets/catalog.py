import json
from ssl import SSLCertVerificationError
from typing import List, Optional

import gcsfs
from pydantic import BaseModel


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
    datasets: List[DatasetMetadata]

    @staticmethod
    def load() -> "Catalog":
        gcs_file_system = gcsfs.GCSFileSystem(token="anon")
        gcs_json_path = "gs://pinecone-datasets-dev/catalog.json"
        try:
            with gcs_file_system.open(gcs_json_path) as f:
                _catalog = json.load(f)
                return Catalog.parse_obj({"datasets": _catalog})
        except SSLCertVerificationError:
            raise ValueError("There is an Issue with loading the public catalog")

    def list_datasets(self) -> List[str]:
        return [dataset.name for dataset in self.datasets]
