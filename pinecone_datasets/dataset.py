import sys
import glob
import os
import json
from functools import cached_property
from typing import Any, Generator, Iterator, List, Union, Dict
import warnings
from urllib.parse import urlparse

import gcsfs
from pydantic import ValidationError
import s3fs
import polars as pl
import pandas as pd
import pyarrow.parquet as pq

from pinecone_datasets import cfg
from pinecone_datasets.catalog import DatasetMetadata
from pinecone_datasets.fs import get_cloud_fs


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
    def from_pandas(cls, documents: pd.DataFrame, queries: pd.DataFrame, **kwargs):
        """
        Create a Dataset object from a pandas DataFrame

        Args:
            documents (pd.DataFrame): a pandas DataFrame containing the documents
            queries (pd.DataFrame): a pandas DataFrame containing the queries

        Returns:
            Dataset: a Dataset object
        """
        clazz = cls
        clazz.documents = documents
        clazz.queries = queries

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

    def _safe_read_from_path(self, data_type: str) -> Union[pl.DataFrame, pd.DataFrame]:
        read_path_str = os.path.join(self._dataset_path, data_type, "*.parquet")
        read_path = self._fs.glob(read_path_str)
        if self._is_datatype_exists(data_type):
            dataset = pq.ParquetDataset(read_path, filesystem=self._fs)
            dataset_schema_names = dataset.schema.names
            for column in getattr(self._config.Schema.Names, data_type):
                print(getattr(self._config.Schema.Names, data_type))
                print(dataset_schema_names)
                if column not in dataset_schema_names:
                    raise ValueError(
                        f"error, file is not matching Pinecone Datasets Schmea: {column} not found"
                    )
            try:
                df = dataset.read_pandas().to_pandas()
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

    def __getitem__(self, key: str) -> pl.DataFrame:
        if key in ["documents", "queries"]:
            return getattr(self, key)
        else:
            raise KeyError("Dataset does not have key: {}".format(key))

    def __len__(self) -> int:
        return self.documents.shape[0]

    @cached_property
    def documents(self) -> Union[pl.DataFrame, pd.DataFrame]:
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
    def queries(self) -> Union[pl.DataFrame, pd.DataFrame]:
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
    def metadata(self) -> Union[pl.DataFrame, pd.DataFrame]:
        return self._load_metadata()

    def head(self, n: int = 5) -> Union[pl.DataFrame, pd.DataFrame]:
        return self.documents.head(n)


    def save_to_catalog(self, dataset_id: str):
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

        # save documents
        documents_path = os.path.join(dataset_path, "documents")
        self.documents.to_parquet(documents_path, engine="pyarrow", index=False)

        # save queries
        queries_path = os.path.join(dataset_path, "queries")
        self.queries.to_parquet(queries_path, engine="pyarrow", index=False)

        # save metadata
        metadata = self.metadata



    def save_to_path(self, dataset_path: str):
        """
        Saves the dataset to a local or cloud storage path.
        """
        raise NotImplementedError
