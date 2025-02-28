import os
import json
import warnings
import logging

from .fs import get_cloud_fs, CloudOrLocalFS
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
else:
    pd = None

logger = logging.getLogger(__name__)


class DatasetFSWriter:
    @staticmethod
    def write_dataset(dataset_path: str, dataset: "Dataset", **kwargs):
        """
        Saves the dataset to a local or cloud storage path.
        """
        fs = get_cloud_fs(dataset_path, **kwargs)
        logger.debug(f"writing dataset {dataset.metadata.name} to {dataset_path}")
        DatasetFSWriter._write_documents(fs, dataset_path, dataset)
        DatasetFSWriter._write_queries(fs, dataset_path, dataset)
        DatasetFSWriter._write_metadata(fs, dataset_path, dataset)

    @staticmethod
    def _write_documents(fs: CloudOrLocalFS, dataset_path: str, dataset: "Dataset"):
        documents_path = os.path.join(dataset_path, "documents")
        fs.makedirs(documents_path, exist_ok=True)

        documents_metadta_copy = dataset.documents["metadata"].copy()
        try:
            logger.debug(
                f"writing dataset {dataset.metadata.name} documents to {documents_path}"
            )
            dataset.documents["metadata"] = dataset.documents["metadata"].apply(
                DatasetFSWriter._convert_metadata_from_dict_to_json
            )
            dataset.documents.to_parquet(
                os.path.join(documents_path, "part-0.parquet"),
                engine="pyarrow",
                index=False,
                filesystem=fs,
            )
        finally:
            dataset.documents["metadata"] = documents_metadta_copy

    @staticmethod
    def _write_queries(fs: CloudOrLocalFS, dataset_path: str, dataset: "Dataset"):
        if dataset.queries.empty:
            warnings.warn("Queries are empty, not saving queries")
        else:
            queries_path = os.path.join(dataset_path, "queries")
            logger.debug(
                f"writing dataset {dataset.metadata.name} queries to {queries_path}"
            )
            fs.makedirs(queries_path, exist_ok=True)
            queries_filter_copy = dataset.queries["filter"].copy()
            try:
                dataset.queries["filter"] = dataset.queries["filter"].apply(
                    DatasetFSWriter._convert_metadata_from_dict_to_json
                )
                dataset.queries.to_parquet(
                    os.path.join(queries_path, "part-0.parquet"),
                    engine="pyarrow",
                    index=False,
                    filesystem=fs,
                )
            finally:
                dataset.queries["filter"] = queries_filter_copy

    @staticmethod
    def _write_metadata(fs: CloudOrLocalFS, dataset_path: str, dataset: "Dataset"):
        metadata_path = os.path.join(dataset_path, "metadata.json")
        logger.debug(
            f"writing dataset {dataset.metadata.name} metadata to {metadata_path}"
        )
        with fs.open(metadata_path, "w") as f:
            json.dump(dataset.metadata.model_dump(), f)

    @staticmethod
    def _convert_metadata_from_dict_to_json(metadata: Optional[dict]) -> str:
        import pandas as pd

        if pd.isna(metadata):
            return None
        if metadata and not isinstance(metadata, dict):
            raise TypeError(
                f"metadata must be a dict but its {type(metadata)} meta = {metadata}"
            )
        return json.dumps(metadata, ensure_ascii=False)
