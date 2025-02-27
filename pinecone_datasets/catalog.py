import warnings
import os
import json
from ssl import SSLCertVerificationError
from typing import List, Optional, Union
import s3fs
import gcsfs
from pydantic import BaseModel, ValidationError, Field
import pandas as pd

from .cfg import Storage
from .fs import get_cloud_fs
from .dataset import Dataset
from .dataset_fswriter import DatasetFSWriter
from .dataset_metadata import DatasetMetadata


class Catalog(BaseModel):
    def __init__(self, base_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if base_path is None:
            self.base_path = os.environ.get(
                "DATASETS_CATALOG_BASEPATH", Storage.endpoint
            )
        else:
            self.base_path = base_path

    base_path: str = Field(default=None)
    datasets: List[DatasetMetadata] = []

    def load(self, **kwargs) -> "Catalog":
        """Loads the catalog from the cloud storage path."""
        fs = get_cloud_fs(self.base_path, **kwargs)
        if not fs:
            raise ValueError(
                "Public datasets are only supported on cloud storage, with valid s3:// or gs:// paths"
            )
        collected_datasets = []
        try:
            for f in fs.listdir(self.base_path):
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
                                    f"metadata file for dataset: {f} is not valid, skipping"
                                )
                    except FileNotFoundError:
                        pass
            self.datasets = collected_datasets
            return self
        except SSLCertVerificationError:
            raise ValueError("There is an Issue with loading the public catalog")

    def list_datasets(self, as_df: bool) -> Union[List[str], pd.DataFrame]:
        if as_df:
            return pd.DataFrame([ds.model_dump() for ds in self.datasets])
        else:
            return [dataset.name for dataset in self.datasets]

    def load_dataset(self, dataset_id: str, **kwargs) -> "Dataset":
        """Loads the dataset from the cloud storage path."""
        self.load(**kwargs)
        for ds in self.datasets:
            if ds.name == dataset_id:
                return Dataset.from_catalog(dataset_id, **kwargs)
        raise FileNotFoundError(f"Dataset {dataset_id} not found in catalog")

    def save_dataset(
        self,
        dataset: "Dataset",
        **kwargs,
    ):
        """
        Saves the dataset to the public catalog.
        """
        DatasetFSWriter.write_dataset(self.base_path, dataset, **kwargs)
