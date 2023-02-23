import sys
import glob
import os
from typing import Any, Generator, Iterator, List, Union, Dict
import warnings

import gcsfs
import s3fs
import polars as pl
import pandas as pd
import pyarrow.parquet as pq

from pinecone_datasets import cfg


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
    def __init__(
        self, dataset_id: str = "", base_path: str = "", engine: str = "pandas"
    ) -> None:
        """
        Dataset class to load and query datasets from the Pinecone Datasets catalog.

        Args:
            dataset_id (str, optional): The dataset id. Defaults to "".
            base_path (str, optional): The path to the dataset. Defaults to "". by default, datasets will look for datasets at path: {base_path}/{dataset_id}/
            engine (str, optional): The engine to use for loading the dataset. Options are ['polars', 'pandas']. Defaults to 'pandas'.


        Examples:
            from pinecone_datasets import Dataset
            dataset = Dataset("dataset_name")
            for doc in dataset.iter_documents(batch_size=100):
                index.upsert(doc)
            for query in dataset.iter_queries(batch_size):
                results = index.search(query)
                # do something with the results
            # or
            dataset.documents # returns a pandas/polars DataFrame
            dataset.queries # returns a pandas/polars DataFrame

        """
        self._documents: pl.DataFrame = None
        self._queries: pl.DataFrame = None
        self._config = cfg
        self._base_path = base_path if base_path else self._config.Storage.base_path
        self._engine = engine
        self._fs = None
        if (
            self._base_path.startswith("gs://")
            or "storage.googleapis.com" in self._base_path
        ):
            self._fs = gcsfs.GCSFileSystem(token="anon")
        elif (
            self._base_path.startswith("s3://") or "s3.amazonaws.com" in self._base_path
        ):
            self._fs = s3fs.S3FileSystem()
        if dataset_id:
            self._load(dataset_id)

    def _create_path(self, dataset_id: str) -> str:
        path = os.path.join(self._base_path, f"{dataset_id}")
        return path

    def _is_datatype_exists(self, data_type: str, dataset_id: str) -> bool:
        if self._fs:
            # TODO: implement for s3 and gcs
            return True
        else:
            return os.path.exists(
                os.path.join(self._create_path(dataset_id), data_type)
            )

    def _safe_read_from_path(
        self, data_type: str, dataset_id: str, enforced_schema: Dict[str, Any]
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        read_path_str = os.path.join(
            self._create_path(dataset_id), data_type, "*.parquet"
        )
        read_path = (
            self._fs.glob(read_path_str) if self._fs else glob.glob(read_path_str)
        )
        if self._is_datatype_exists(data_type, dataset_id):
            dataset = pq.ParquetDataset(read_path, filesystem=self._fs)
            try:
                if self._engine == "pandas":
                    df = dataset.read_pandas().to_pandas()
                elif self._engine == "polars":
                    df = pl.from_arrow(dataset.read(), schema_overrides=enforced_schema)
                else:
                    raise ValueError("engine must be one of ['pandas', 'polars']")
                return df
            except pl.PanicException as pe:
                msg = f"error, file is not matching Pinecone Datasets Schmea: {pe}"
                raise RuntimeError(msg)
            except Exception as e:
                print("error, no exception: {}".format(e), file=sys.stderr)
                raise (e)
        else:
            warnings.warn(
                "WARNING: No data found at: {}. Returning empty DF".format(
                    read_path_str
                ),
                UserWarning,
                stacklevel=0
            )
            if self._engine == "pandas":
                return pd.DataFrame()
            elif self._engine == "polars":
                return pl.DataFrame()
            else:
                raise ValueError("engine must be one of ['pandas', 'polars']")

    def _load(self, dataset_id: str) -> None:
        self._documents = self._safe_read_from_path(
            "documents", dataset_id, self._config.Schema.documents
        )
        self._queries = self._safe_read_from_path(
            "queries", dataset_id, self._config.Schema.queries
        )

    def __getitem__(self, key: str) -> pl.DataFrame:
        if key in ["documents", "queries"]:
            return getattr(self, key)
        else:
            raise KeyError("Dataset does not have key: {}".format(key))

    def __len__(self) -> int:
        return self._documents.shape[0]

    @property
    def documents(self) -> Union[pl.DataFrame, pd.DataFrame]:
        return self._documents

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
            if self._engine == "pandas":
                return iter_pandas_dataframe_slices(
                    self._documents[self._config.Schema.documents_select_columns],
                    batch_size,
                )
            return map(
                lambda x: x.to_dicts(),
                self._documents.select(
                    self._config.Schema.documents_select_columns
                ).iter_slices(n_rows=batch_size),
            )
        else:
            raise ValueError("batch_size must be greater than 0")

    @property
    def queries(self) -> Union[pl.DataFrame, pd.DataFrame]:
        return self._queries

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
        if self._engine == "pandas":
            return iter_pandas_dataframe_single(
                self._queries[self._config.Schema.queries_select_columns]
            )
        else:
            return self._queries.select(
                self._queries[self._config.Schema.queries_select_columns]
            ).iter_rows(named=True)

    def head(self, n: int = 5) -> Union[pl.DataFrame, pd.DataFrame]:
        return self.documents.head(n)
