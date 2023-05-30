from datetime import datetime
import warnings
import os
import json
from ssl import SSLCertVerificationError
from typing import List, Optional, Union, Any, Dict
import s3fs
import gcsfs
from pydantic import BaseModel, ValidationError, Field
import pandas as pd

from pinecone_datasets import cfg
from pinecone_datasets.fs import get_cloud_fs


class DenseModelMetadata(BaseModel):
    name: str
    tokenizer: Optional[str]
    dimension: int


class SparseModelMetdata(BaseModel):
    name: Optional[str]
    tokenizer: Optional[str]


def get_time_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


class DatasetMetadata(BaseModel):
    name: str
    created_at: str
    documents: int
    queries: int
    source: Optional[str]
    license: Optional[str]
    bucket: Optional[str]
    task: Optional[str]
    dense_model: DenseModelMetadata
    sparse_model: Optional[SparseModelMetdata]
    description: Optional[str]
    tags: Optional[List[str]]
    args: Optional[Dict[str, Any]]


class Catalog(BaseModel):
    datasets: List[DatasetMetadata] = []

    @staticmethod
    def load(**kwargs) -> "Catalog":
        public_datasets_base_path = os.environ.get(
            "DATASETS_CATALOG_BASEPATH", cfg.Storage.endpoint
        )
        fs = get_cloud_fs(public_datasets_base_path, **kwargs)
        if not fs:
            raise ValueError(
                "Public datasets are only supported on cloud storage, with valid s3:// or gs:// paths"
            )
        collected_datasets = []
        try:
            for f in fs.listdir(public_datasets_base_path):
                if f["type"] == "directory":
                    try:
                        prefix = "gs" if isinstance(fs, gcsfs.GCSFileSystem) else "s3"
                        with fs.open(f"{prefix}://{f['name']}/metadata.json") as f:
                            try:
                                this_dataset_json = json.load(f)
                            except json.JSONDecodeError:
                                warnings.warn(
                                    f"Not a JSON: Invalid metadata.json for {f['name']}, skipping"
                                )
                            try:
                                this_dataset = DatasetMetadata(**this_dataset_json)
                                collected_datasets.append(this_dataset)
                            except ValidationError:
                                warnings.warn(
                                    f"metadata file for dataset: {f['name']} is not valid, skipping"
                                )
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
