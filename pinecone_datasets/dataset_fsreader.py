import os
import json
import logging
import warnings
from typing import Literal, Optional

import pandas as pd
import pyarrow.parquet as pq
from .tqdm import tqdm

from .cfg import Schema
from .dataset_metadata import DatasetMetadata
from .fs import CloudOrLocalFS

logger = logging.getLogger(__name__)


class DatasetFSReader:
    @staticmethod
    def read_documents(fs: CloudOrLocalFS, dataset_path: str) -> pd.DataFrame:
        logger.debug(f"reading documents from {dataset_path}")
        df = DatasetFSReader._safe_read_from_path(fs, dataset_path, "documents")

        # metadata supposed to be a dict [if legacy] or string
        df["metadata"] = df["metadata"].apply(
            DatasetFSReader._convert_metadata_from_json_to_dict
        )
        return df

    @staticmethod
    def read_queries(fs: CloudOrLocalFS, dataset_path: str) -> pd.DataFrame:
        logger.debug(f"reading queries from {dataset_path}")
        df = DatasetFSReader._safe_read_from_path(fs, dataset_path, "queries")

        # filter supposed to be a dict [if legacy] or string
        df["filter"] = df["filter"].apply(
            DatasetFSReader._convert_metadata_from_json_to_dict
        )

        return df

    @staticmethod
    def read_metadata(fs: CloudOrLocalFS, dataset_path: str) -> DatasetMetadata:
        logger.debug(f"reading metadata from {dataset_path}")
        with fs.open(os.path.join(dataset_path, "metadata.json"), "rb") as f:
            metadata = json.load(f)
        return DatasetMetadata(**metadata)

    @staticmethod
    def _convert_metadata_from_json_to_dict(metadata: Optional[str] = None) -> dict:
        if metadata is None:
            return None
        elif isinstance(metadata, dict):
            return metadata
        elif isinstance(metadata, str):
            return json.loads(metadata)
        else:
            raise TypeError("metadata must be a string or dict")

    @staticmethod
    def _does_datatype_exist(
        fs: CloudOrLocalFS,
        dataset_path: str,
        data_type: Literal["documents", "queries"],
    ) -> bool:
        return fs.exists(os.path.join(dataset_path, data_type))

    @staticmethod
    def _safe_read_from_path(
        fs: CloudOrLocalFS,
        dataset_path: str,
        data_type: Literal["documents", "queries"],
    ) -> pd.DataFrame:
        read_path_str = os.path.join(dataset_path, data_type, "*.parquet")
        read_path = fs.glob(read_path_str)
        if DatasetFSReader._does_datatype_exist(fs, dataset_path, data_type):
            # First, collect all the dataframes
            dfs = []
            for path in tqdm(read_path, desc=f"Loading {data_type} parquet files"):
                piece = pq.read_pandas(path, filesystem=fs)
                df_piece = piece.to_pandas()
                dfs.append(df_piece)

            if not dfs:
                raise ValueError(f"No parquet files found in {read_path_str}")

            # Combine all dataframes
            df = pd.concat(dfs, ignore_index=True)

            # Validate schema
            dataset_schema_names = df.columns.tolist()
            columns_to_null = []
            columns_not_null = []
            for column_name, is_nullable, null_value in getattr(
                Schema.Names, data_type
            ):
                if column_name not in dataset_schema_names and not is_nullable:
                    raise ValueError(
                        f"error, file is not matching Pinecone Datasets Schema: {column_name} not found"
                    )
                elif column_name not in dataset_schema_names and is_nullable:
                    columns_to_null.append((column_name, null_value))
                else:
                    columns_not_null.append(column_name)

            # Add null columns if needed
            for column_name, null_value in columns_to_null:
                df[column_name] = null_value

            return df[columns_not_null + [col for col, _ in columns_to_null]]

        else:
            warnings.warn(
                "WARNING: No data found at: {}. Returning empty dataframe".format(
                    read_path_str
                ),
                UserWarning,
                stacklevel=0,
            )
            return pd.DataFrame(
                columns=[col[0] for col in getattr(Schema.Names, data_type)]
            )
