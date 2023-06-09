import sys
import glob
import os
import json
from functools import cached_property
from typing import Any, Generator, Iterator, List, Union, Dict, Optional, Tuple
import warnings
from urllib.parse import urlparse

import gcsfs
from pydantic import ValidationError
import s3fs

# import polars as pl
import pandas as pd
import pyarrow.parquet as pq

from pinecone_datasets import cfg
from pinecone_datasets.catalog import DatasetMetadata
from pinecone_datasets.fs import get_cloud_fs, LocalFileSystem


def iter_pandas_dataframe_slices(
    df: pd.DataFrame, batch_size=1
) -> Generator[List[Dict[str, Any]], None, None]:
    for i in range(0, len(df), batch_size):
        yield df.iloc[i : i + batch_size].to_dict(orient="records")


def iter_pandas_dataframe_single(
    df: pd.DataFrame,
) -> Generator[Dict[str, Any], None, None]:
    for i in range(0, len(df), 1):
        yield df.iloc[i : i + 1].to_dict(orient="records")[0]


class Dataset(object):
    @classmethod
    def from_path(cls, dataset_path, **kwargs):
        """
        Create a Dataset object from local or cloud storage
        Args:
            dataset_path (str): a path to a local or cloud storage path containing a valid dataset.

        Keyword Args:
            engine (str): the engine to use for loading the dataset. Options are ['polars', 'pandas']. Defaults to 'pandas'.

        Returns:
            Dataset: a Dataset object
        """
        return cls(dataset_path=dataset_path, **kwargs)

    @classmethod
    def from_catalog(cls, dataset_id, catalog_base_path: str = "", **kwargs):
        """
        Load a dataset from Pinecone's Datasets catalog, or from your own endpoint.

        Args:
            dataset_id (str): the id of the dataset to load within a catalog
            catalog_base_path (str): the catalog's base path. Defaults to DATASETS_CATALOG_BASEPATH environment variable.
                                     If neither are set, will use Pinecone's public catalog.

        Returns:
            Dataset: a Dataset object
        """
        catalog_base_path = (
            catalog_base_path
            if catalog_base_path
            else os.environ.get("DATASETS_CATALOG_BASEPATH", cfg.Storage.endpoint)
        )
        dataset_path = os.path.join(catalog_base_path, f"{dataset_id}")
        return cls(dataset_path=dataset_path, **kwargs)

    @classmethod
    def from_pandas(
        cls,
        documents: pd.DataFrame,
        metadata: DatasetMetadata,
        documents_column_mapping: Optional[Dict] = None,
        queries: Optional[pd.DataFrame] = None,
        queries_column_mapping: Optional[Dict] = None,
        **kwargs,
    ) -> "Dataset":
        """
        Create a Dataset object from a pandas DataFrame

        Args:
            documents (pd.DataFrame): a pandas DataFrame containing the documents
            documents_column_mapping (Dict): a dictionary mapping the columns of the documents DataFrame to the Pinecone Datasets Schema
            queries (pd.DataFrame): a pandas DataFrame containing the queries
            queries_column_mapping (Dict): a dictionary mapping the columns of the queries DataFrame to the Pinecone Datasets Schema

        Keyword Args:
            kwargs (Dict): additional arguments to pass to the fsspec constructor

        Returns:
            Dataset: a Dataset object
        """
        clazz = Dataset(dataset_path=os.getcwd(), **kwargs)
        clazz.documents = cls._read_pandas_dataframe(
            documents, documents_column_mapping, cfg.Schema.Names.documents
        )
        clazz.queries = cls._read_pandas_dataframe(
            queries, queries_column_mapping, cfg.Schema.Names.queries
        )
        clazz.metadata = metadata
        return clazz

    @staticmethod
    def _read_pandas_dataframe(
        df: pd.DataFrame, column_mapping: Dict[str, str], schema: List[Tuple[str, bool]]
    ) -> pd.DataFrame:
        """
        Reads a pandas DataFrame and validates it against a schema.

        Args:
            df (pd.DataFrame): the pandas DataFrame to read
            column_mapping (Dict[str, str]): a dictionary mapping the columns of the DataFrame to the Pinecone Datasets Schema (col_name, pinecone_name)
            schema (List[Tuple[str, bool]]): the schema to validate against (column_name, is_nullable)

        Returns:
            pd.DataFrame: the validated, renamed DataFrame
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=[column_name for column_name, _ in schema])
        else:
            if column_mapping is not None:
                df.rename(columns=column_mapping, inplace=True)
            for column_name, is_nullable in schema:
                if column_name not in df.columns and not is_nullable:
                    raise ValueError(
                        f"error, file is not matching Pinecone Datasets Schmea: {column_name} not found"
                    )
                elif column_name not in df.columns and is_nullable:
                    df[column_name] = None
            return df[[column_name for column_name, _ in schema]]

    def __init__(
        self,
        dataset_path: str,
        **kwargs,
    ) -> None:
        """
        Dataset class to load and query datasets from the Pinecone Datasets catalog.
        See `from_path` and `from_dataset_id` for examples on how to load a dataset.

        Examples:
            ```python
            from pinecone_datasets import Dataset
            dataset = Dataset.from_dataset_id("dataset_name")
            # or
            dataset = Dataset.from_path("gs://my-bucket/my-dataset")

            for doc in dataset.iter_documents(batch_size=100):
                index.upsert(doc)
            for query in dataset.iter_queries(batch_size):
                results = index.search(query)
                # do something with the results
            # or
            dataset.documents # returns a pandas/polars DataFrame
            dataset.queries # returns a pandas/polars DataFrame
            ```

        """
        self._config = cfg
        endpoint = urlparse(dataset_path)._replace(path="").geturl()
        self._fs = get_cloud_fs(endpoint, **kwargs)
        self._dataset_path = dataset_path

        if not self._fs.exists(self._dataset_path):
            raise FileNotFoundError(
                "Dataset does not exist. Please check the path or dataset_id"
            )

    def _is_datatype_exists(self, data_type: str) -> bool:
        return self._fs.exists(os.path.join(self._dataset_path, data_type))

    def _safe_read_from_path(self, data_type: str) -> pd.DataFrame:
        read_path_str = os.path.join(self._dataset_path, data_type, "*.parquet")
        read_path = self._fs.glob(read_path_str)
        if self._is_datatype_exists(data_type):
            dataset = pq.ParquetDataset(read_path, filesystem=self._fs)
            dataset_schema_names = dataset.schema.names
            columns_to_null = []
            columns_not_null = []
            for column_name, is_nullable in getattr(
                self._config.Schema.Names, data_type
            ):
                if column_name not in dataset_schema_names and not is_nullable:
                    raise ValueError(
                        f"error, file is not matching Pinecone Datasets Schmea: {column_name} not found"
                    )
                elif column_name not in dataset_schema_names and is_nullable:
                    columns_to_null.append(column_name)
                else:
                    columns_not_null.append(column_name)
            try:
                # TODO: use of the columns_not_null and columns_to_null is only a workaround for proper schema validation and versioning
                df = dataset.read_pandas(columns=columns_not_null).to_pandas()
                for column_name in columns_to_null:
                    df[column_name] = None
                return df
            # TODO: add more specific error handling, explain what is wrong
            except Exception as e:
                print("error, no exception: {}".format(e), file=sys.stderr)
                raise (e)
        else:
            warnings.warn(
                "WARNING: No data found at: {}. Returning empty DF".format(
                    read_path_str
                ),
                UserWarning,
                stacklevel=0,
            )
            return pd.DataFrame(columns=getattr(self._config.Schema.Names, data_type))

    def _load_metadata(self) -> DatasetMetadata:
        with self._fs.open(
            os.path.join(self._dataset_path, "metadata.json"), "rb"
        ) as f:
            metadata = json.load(f)
        try:
            out = DatasetMetadata(**metadata)
            return out
        # TODO: add more specific error handling, explain what is wrong
        except ValidationError as e:
            raise e

    def _save_metadata(self, metadata: DatasetMetadata) -> None:  # pragma: no cover
        with self._fs.open(os.path.join(self._dataset_path, "metadata.json"), "w") as f:
            json.dump(metadata.dict(), f)

    def __getitem__(self, key: str):
        if key in ["documents", "queries"]:
            return getattr(self, key)
        else:
            raise KeyError("Dataset does not have key: {}".format(key))

    def __len__(self) -> int:
        return self.documents.shape[0]

    @cached_property
    def documents(self) -> pd.DataFrame:
        return self._safe_read_from_path("documents")

    def iter_documents(self, batch_size: int = 1) -> Iterator[List[Dict[str, Any]]]:
        """
        Iterates over the documents in the dataset.

        Args:
            batch_size (int, optional): The batch size to use for the iterator. Defaults to 1.

        Returns:
            Iterator[List[Dict[str, Any]]]: An iterator over the documents in the dataset.

        Examples:
            for batch in dataset.iter_documents(batch_size=100):
                index.upsert(batch)
        """
        if isinstance(batch_size, int) and batch_size > 0:
            return iter_pandas_dataframe_slices(
                self.documents[self._config.Schema.documents_select_columns],
                batch_size,
            )
        else:
            raise ValueError("batch_size must be greater than 0")

    @cached_property
    def queries(self) -> pd.DataFrame:
        return self._safe_read_from_path("queries")

    def iter_queries(self) -> Iterator[Dict[str, Any]]:
        """
        Iterates over the queries in the dataset.

        Returns:
            Iterator[Dict[str, Any]]: An iterator over the queries in the dataset.

        Examples:
            for query in dataset.iter_queries():
                results = index.query(**query)
                # do something with the results
        """
        return iter_pandas_dataframe_single(
            self.queries[self._config.Schema.queries_select_columns]
        )

    @cached_property
    def metadata(self) -> pd.DataFrame:
        return self._load_metadata()

    def head(self, n: int = 5) -> pd.DataFrame:
        return self.documents.head(n)

    def save_to_path(self, dataset_path: str, **kwargs):
        """
        Saves the dataset to a local or cloud storage path.
        """
        fs = get_cloud_fs(dataset_path, **kwargs)

        # save documents
        documents_path = os.path.join(dataset_path, "documents")
        fs.makedirs(documents_path, exist_ok=True)
        self.documents.to_parquet(
            os.path.join(documents_path, "part-0.parquet"),
            engine="pyarrow",
            index=False,
            filesystem=fs,
        )
        # save queries
        if not self.queries.empty:
            queries_path = os.path.join(dataset_path, "queries")
            fs.makedirs(queries_path, exist_ok=True)
            self.queries.to_parquet(
                os.path.join(queries_path, "part-0.parquet"),
                engine="pyarrow",
                index=False,
                filesystem=fs,
            )
        else:
            warnings.warn("Queries are empty, not saving queries")

        # save metadata
        with fs.open(os.path.join(dataset_path, "metadata.json"), "w") as f:
            json.dump(self.metadata.dict(), f)

    def save_to_catalog(self, dataset_id: str, catalog_base_path: str = "", **kwargs):
        """
        Saves the dataset to the public catalog.
        """

        # TODO: duplicated code

        catalog_base_path = (
            catalog_base_path
            if catalog_base_path
            else os.environ.get("DATASETS_CATALOG_BASEPATH", cfg.Storage.endpoint)
        )
        dataset_path = os.path.join(catalog_base_path, f"{dataset_id}")
        self.save_to_path(dataset_path, **kwargs)
