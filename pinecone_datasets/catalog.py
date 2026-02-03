import json
import logging
import os
import warnings
from typing import TYPE_CHECKING, Optional, Union

from pydantic import BaseModel, Field, ValidationError

from .cfg import Storage
from .dataset import Dataset
from .dataset_fswriter import DatasetFSWriter
from .dataset_metadata import DatasetMetadata
from .fs import get_cloud_fs

if TYPE_CHECKING:
    import pandas as pd
else:
    pd = None


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
    datasets: list[DatasetMetadata] = Field(default_factory=list)

    def load(self, **kwargs) -> "Catalog":
        """Loads metadata about all datasets from the catalog."""
        fs = get_cloud_fs(self.base_path, **kwargs)
        collected_datasets = []

        metadata_files_glob_path = os.path.join(self.base_path, "*", "metadata.json")
        for metadata_path in fs.glob(metadata_files_glob_path):
            with fs.open(metadata_path) as f:
                try:
                    this_dataset_json = json.load(f)
                except json.JSONDecodeError:
                    warnings.warn(
                        f"Not a JSON: Invalid metadata.json for {metadata_path}, skipping"
                    )
                    continue

                try:
                    this_dataset = DatasetMetadata(**this_dataset_json)
                    collected_datasets.append(this_dataset)
                except ValidationError as e:
                    warnings.warn(
                        f"metadata file for dataset: {metadata_path} is not valid, skipping: {e}"
                    )
                    continue

        self.datasets = collected_datasets
        logger.info(f"Loaded {len(self.datasets)} datasets from {self.base_path}")
        return self

    def list_datasets(self, as_df: bool) -> Union[list[str], "pd.DataFrame"]:
        """Lists all datasets in the catalog."""
        if self.datasets is None or len(self.datasets) == 0:
            self.load()

        import pandas as pd

        if as_df:
            return pd.DataFrame([ds.model_dump() for ds in self.datasets])
        else:
            return [dataset.name for dataset in self.datasets]

    def load_dataset(self, dataset_id: str, **kwargs) -> "Dataset":
        """Loads the dataset from the catalog."""
        ds_path = os.path.join(str(self.base_path), dataset_id)
        return Dataset.from_path(dataset_path=ds_path, **kwargs)

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
