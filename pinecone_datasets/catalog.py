import json
from ssl import SSLCertVerificationError
from typing import List, Optional, Union

import gcsfs
from pydantic import BaseModel
import pandas as pd


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
    documents: int
    queries: int
    source: Optional[str]
    bucket: str
    task: str
    dense_model: DenseModelMetadata
    sparse_model: Optional[SparseModelMetdata]


class Catalog(BaseModel):
    datasets: List[DatasetMetadata] = []

    @staticmethod
    def load() -> "Catalog":
        gcs_file_system = gcsfs.GCSFileSystem(token="anon")
        gcs_publid_datasets_base_path = "gs://pinecone-datasets-dev"
        gcs_json_path = f"{gcs_publid_datasets_base_path}/catalog.json"
        collected_datasets = []
        try: 
            for f in  gcs_file_system.listdir(gcs_publid_datasets_base_path):
                if f["type"] == "directory":
                    try:
                        with gcs_file_system.open(f"gs://{f['name']}/metadata.json") as f:
                            this_dataset = json.load(f)
                            collected_datasets.append(this_dataset)
                    except FileNotFoundError:
                        pass
            return Catalog(datasets=collected_datasets)
        except SSLCertVerificationError:
            raise ValueError("There is an Issue with loading the public catalog")

    def list_datasets(self, as_df: bool) -> Union[List[str], pd.DataFrame]:
        if as_df:
            return pd.DataFrame([ds.dict() for ds in self.datasets])
        else:
            return [dataset.name for dataset in self.datasets]
