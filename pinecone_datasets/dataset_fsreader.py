import json
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq

from .cfg import Cache, Schema
from .dataset_metadata import DatasetMetadata
from .fs import CloudOrLocalFS, get_cached_path, is_cloud_path
from .retry import create_cloud_storage_retry_decorator
from .tqdm import tqdm

logger = logging.getLogger(__name__)

retry_decorator = create_cloud_storage_retry_decorator()


class DatasetFSReader:
    @staticmethod
    @retry_decorator
    def read_documents(fs: CloudOrLocalFS, dataset_path: str) -> pd.DataFrame:
        logger.debug(f"reading documents from {dataset_path}")
        df = DatasetFSReader._safe_read_from_path(fs, dataset_path, "documents")

        # metadata supposed to be a dict [if legacy] or string
        df["metadata"] = df["metadata"].apply(
            DatasetFSReader._convert_metadata_from_json_to_dict
        )
        return df

    @staticmethod
    @retry_decorator
    def read_queries(fs: CloudOrLocalFS, dataset_path: str) -> pd.DataFrame:
        logger.debug(f"reading queries from {dataset_path}")
        df = DatasetFSReader._safe_read_from_path(fs, dataset_path, "queries")

        # filter supposed to be a dict [if legacy] or string
        df["filter"] = df["filter"].apply(
            DatasetFSReader._convert_metadata_from_json_to_dict
        )

        return df

    @staticmethod
    @retry_decorator
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
    def _download_and_read_parquet(
        path: str,
        fs: CloudOrLocalFS,
        use_cache: bool,
        protocol: Optional[str],
    ) -> Tuple[int, pd.DataFrame]:
        """
        Download (if needed) and read a single parquet file.

        Args:
            path: Path to the parquet file
            fs: Filesystem object
            use_cache: Whether to use caching for this file
            protocol: Protocol prefix (gs:// or s3://) if applicable

        Returns:
            Tuple of (file_index, dataframe) where file_index is from the path
        """
        if use_cache and protocol:
            # Reconstruct full URL if path doesn't have protocol
            if not path.startswith(protocol):
                full_path = f"{protocol}{path}"
            else:
                full_path = path
            # Download to cache and read from local path
            local_path = get_cached_path(full_path, fs)
            piece = pq.read_pandas(local_path)
        else:
            # Read directly from filesystem
            piece = pq.read_pandas(path, filesystem=fs)

        df_piece = piece.to_pandas()
        # Extract index from path for proper ordering (handles paths like "documents/0000.parquet")
        try:
            filename = os.path.basename(path)
            file_index = int(os.path.splitext(filename)[0])
        except (ValueError, AttributeError):
            # If we can't extract an index, use hash of path for consistent ordering
            file_index = hash(path)

        return (file_index, df_piece)

    @staticmethod
    def _safe_read_from_path(
        fs: CloudOrLocalFS,
        dataset_path: str,
        data_type: Literal["documents", "queries"],
    ) -> pd.DataFrame:
        read_path_str = os.path.join(dataset_path, data_type, "*.parquet")
        read_path = fs.glob(read_path_str)
        if DatasetFSReader._does_datatype_exist(fs, dataset_path, data_type):
            # Determine if we should use cache based on dataset_path
            use_cache_for_dataset = is_cloud_path(dataset_path)

            # Determine protocol prefix for reconstructing full URLs
            protocol = None
            if dataset_path.startswith("gs://"):
                protocol = "gs://"
            elif dataset_path.startswith("s3://"):
                protocol = "s3://"
            elif dataset_path.startswith("https://storage.googleapis.com/"):
                protocol = "gs://"
            elif dataset_path.startswith("https://s3.amazonaws.com/"):
                protocol = "s3://"

            # Collect all dataframes using parallel downloads
            num_files = len(read_path)
            max_workers = (
                min(Cache.max_parallel_downloads, num_files) if num_files > 1 else 1
            )

            dfs_with_index = []

            if max_workers == 1:
                # Serial processing for single file or when max_workers=1
                for path in tqdm(read_path, desc=f"Loading {data_type}"):
                    file_index, df_piece = DatasetFSReader._download_and_read_parquet(
                        path, fs, use_cache_for_dataset, protocol
                    )
                    dfs_with_index.append((file_index, df_piece))
            else:
                # Parallel processing for multiple files
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all download tasks
                    future_to_path = {
                        executor.submit(
                            DatasetFSReader._download_and_read_parquet,
                            path,
                            fs,
                            use_cache_for_dataset,
                            protocol,
                        ): path
                        for path in read_path
                    }

                    # Collect results as they complete with progress bar
                    with tqdm(total=num_files, desc=f"Loading {data_type}") as pbar:
                        for future in as_completed(future_to_path):
                            path = future_to_path[future]
                            try:
                                file_index, df_piece = future.result()
                                dfs_with_index.append((file_index, df_piece))
                            except Exception as e:
                                logger.error(f"Failed to load {path}: {e}")
                                raise
                            finally:
                                pbar.update(1)

            # Sort by file index to maintain consistent ordering
            dfs_with_index.sort(key=lambda x: x[0])
            dfs = [df for _, df in dfs_with_index]

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
                f"WARNING: No data found at: {read_path_str}. Returning empty dataframe",
                UserWarning,
                stacklevel=0,
            )
            return pd.DataFrame(
                columns=[col[0] for col in getattr(Schema.Names, data_type)]
            )
