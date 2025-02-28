import warnings
import os
import json
from ssl import SSLCertVerificationError
from typing import List, Optional, Union
import s3fs
import gcsfs
from fsspec.implementations.local import LocalFileSystem

import logging
from pydantic import BaseModel, ValidationError, Field
import pandas as pd

from .cfg import Storage
from .fs import get_cloud_fs
from .dataset import Dataset
from .dataset_fswriter import DatasetFSWriter
from .dataset_metadata import DatasetMetadata

logger = logging.getLogger(__name__)


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
    datasets: List[DatasetMetadata] = Field(default_factory=list)

    def load(self, **kwargs) -> "Catalog":
        """Loads metadata about all datasets from the catalog."""
        fs = get_cloud_fs(self.base_path, **kwargs)
        collected_datasets = []
        try:
            for f in fs.listdir(self.base_path):
                if f["type"] == "directory":
                    try:
                        if isinstance(fs, LocalFileSystem):
                            metadata_path = f"{f['name']}/metadata.json"
                        elif isinstance(fs, gcsfs.GCSFileSystem):
                            metadata_path = f"gs://{f['name']}/metadata.json"
                        elif isinstance(fs, s3fs.S3FileSystem):
                            metadata_path = f"s3://{f['name']}/metadata.json"
                        else:
                            raise ValueError(f"Unsupported filesystem: {type(fs)}")

                        with fs.open(metadata_path) as f:
                            try:
                                this_dataset_json = json.load(f)
                            except json.JSONDecodeError:
                                warnings.warn(
                                    f"Not a JSON: Invalid metadata.json for {f['name']}, skipping"
                                )
                                continue
                            try:
                                this_dataset = DatasetMetadata(**this_dataset_json)
                                collected_datasets.append(this_dataset)
                            except ValidationError:
                                warnings.warn(
                                    f"metadata file for dataset: {f['name']} is not valid, skipping"
                                )
                                continue
                    except FileNotFoundError:
                        pass
            self.datasets = collected_datasets
            logger.info(f"Loaded {len(self.datasets)} datasets from {self.base_path}")
            logger.debug(
                f"Datasets in {self.base_path}: {self.list_datasets(as_df=False)}"
            )
            return self
        except SSLCertVerificationError:
            raise ValueError("There is an Issue with loading the public catalog")

    def list_datasets(self, as_df: bool) -> Union[List[str], pd.DataFrame]:
        """Lists all datasets in the catalog."""
        if as_df:
            return pd.DataFrame([ds.model_dump() for ds in self.datasets])
        else:
            return [dataset.name for dataset in self.datasets]

    def load_dataset(self, dataset_id: str, **kwargs) -> "Dataset":
        """Loads the dataset from the catalog."""
        self.load(**kwargs)
        for ds in self.datasets:
            if ds.name == dataset_id:
                ds_path = os.path.join(self.base_path, dataset_id)
                logger.info(f"Loading dataset {dataset_id} from {ds_path}")
                return Dataset.from_path(dataset_path=ds_path, **kwargs)
        raise FileNotFoundError(
            f"Dataset {dataset_id} not found in catalog at {self.base_path}"
        )

    def save_dataset(
        self,
        dataset: "Dataset",
        **kwargs,
    ):
        """
        Save a dataset to the catalog.
        """
        ds_path = os.path.join(self.base_path, dataset.metadata.name)
        DatasetFSWriter.write_dataset(dataset_path=ds_path, dataset=dataset, **kwargs)
        logger.info(f"Saved dataset {dataset.metadata.name} to {ds_path}")
