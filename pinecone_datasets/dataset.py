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
    def __init__(
        self,
        dataset_id: str = "",
        dataset_path: str = "",
        endpoint: str = "",
        engine: str = "pandas",
        **kwargs,
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
        if dataset_id and dataset_path:
            raise ValueError(
                "dataset_id and dataset_path cannot be provided at the same time"
            )

        self._config = cfg

        if endpoint:
            self._endpoint = endpoint
        elif dataset_path:
            self._endpoint = urlparse(dataset_path)._replace(path="").geturl()
        else:
            self._endpoint = os.environ.get(
                "PINECONE_DATASETS_EDNPOINT", self._config.Storage.endpoint
            )

        self._engine = engine
        self._fs = get_cloud_fs(self._endpoint, **kwargs)
        if dataset_id:
            self._dataset_path = self._create_path(dataset_id)
        elif dataset_path:
            self._dataset_path = dataset_path
        else:
            raise ValueError("Either dataset_id or dataset_path must be provided")

    def _create_path(self, dataset_id: str) -> str:
        path = os.path.join(self._endpoint, f"{dataset_id}")
        return path

    def _is_datatype_exists(self, data_type: str) -> bool:
        if self._fs:
            key = os.path.join(self._dataset_path, data_type).split("//")[-1]
            for obj in self._fs.ls(self._dataset_path):
                if obj == key:
                    return True
            return False
        else:
            return os.path.exists(
                os.path.join(self._dataset_path, data_type)
            )

    def _safe_read_from_path(
        self, data_type: str, enforced_schema: Dict[str, Any]
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        read_path_str = os.path.join(
            self._dataset_path, data_type, "*.parquet"
        )
        read_path = (
            self._fs.glob(read_path_str) if self._fs else glob.glob(read_path_str)
        )
        if self._is_datatype_exists(data_type):
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
                stacklevel=0,
            )
            if self._engine == "pandas":
                return pd.DataFrame()
            elif self._engine == "polars":
                return pl.DataFrame()
            else:
                raise ValueError("engine must be one of ['pandas', 'polars']")

    def _load_metadata(self) -> None:
        if self._fs:
            with self._fs.open(
                os.path.join(self._dataset_path, "metadata.json"), "rb"
            ) as f:
                metadata = json.load(f)
        else:
            with open(
                os.path.join(self._dataset_path, "metadata.json"), "rb"
            ) as f:
                metadata = json.load(f)
        try:
            out = DatasetMetadata(**metadata)
            return out
        except ValidationError as e:
            raise e

    def _save_metadata(
        self, dataset_id: str, metadata: DatasetMetadata
    ) -> None:  # pragma: no cover
        if self._fs:
            with self._fs.open(
                os.path.join(self._dataset_path, "metadata.json"), "w"
            ) as f:
                json.dump(metadata.dict(), f)
        else:
            with open(
                os.path.join(self._dataset_path, "metadata.json"), "w"
            ) as f:
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
        return self._safe_read_from_path(
            "documents", self._config.Schema.documents
        )

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
                    self.documents[self._config.Schema.documents_select_columns],
                    batch_size,
                )
            return map(
                lambda x: x.to_dicts(),
                self.documents.select(
                    self._config.Schema.documents_select_columns
                ).iter_slices(n_rows=batch_size),
            )
        else:
            raise ValueError("batch_size must be greater than 0")

    @cached_property
    def queries(self) -> Union[pl.DataFrame, pd.DataFrame]:
        return self._safe_read_from_path(
            "queries", self._config.Schema.documents
        )

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
                self.queries[self._config.Schema.queries_select_columns]
            )
        else:
            return self.queries.select(
                self.queries[self._config.Schema.queries_select_columns]
            ).iter_rows(named=True)

    @cached_property
    def metadata(self) -> Union[pl.DataFrame, pd.DataFrame]:
        return self._load_metadata()

    def head(self, n: int = 5) -> Union[pl.DataFrame, pd.DataFrame]:
        return self.documents.head(n)

    @classmethod
    def from_path(cls, dataset_path, **kwargs):
        return cls(dataset_path=dataset_path, **kwargs)

    @classmethod
    def from_dataset_id(cls, dataset_id, **kwargs):
        return cls(dataset_id=dataset_id, **kwargs)
