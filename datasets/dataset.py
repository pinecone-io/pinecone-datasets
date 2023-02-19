import sys
import glob
import os
from typing import Any, Generator, Iterator, List, Union, Dict

import gcsfs
import s3fs
import polars as pl
import pandas as pd
import pyarrow.parquet as pq

from datasets import cfg


def iter_pandas_dataframe_slices(df: pd.DataFrame, batch_size=1) -> Generator[List[Dict[str, Any]], None, None]:
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i+batch_size].to_dict(orient="records")


def iter_pandas_dataframe_single(df: pd.DataFrame) -> Generator[Dict[str, Any], None, None]: 
    for i in range(0, len(df), 1):
        yield df.iloc[i:i+1].to_dict(orient="records")[0]


class Dataset(object):
    def __init__(self, dataset_id: str = None, path: str=None, engine: str = 'pandas') -> None:
        self._documents: pl.DataFrame = None
        self._queries: pl.DataFrame = None
        self._config = cfg
        self._base_path = path if path else self._config.Storage.base_path
        self._engine = engine
        self._fs = None
        if self._base_path.startswith("gs://") or "storage.googleapis.com" in self._base_path: 
            self._fs = gcsfs.GCSFileSystem(token='anon')
        elif self._base_path.startswith("s3://") or "s3.amazonaws.com" in self._base_path:
            self._fs = s3fs.S3FileSystem()
        if dataset_id:
            self._load(dataset_id)

    def _create_path(self, dataset_id: str) -> str:
        path = os.path.join(self._base_path, f"{dataset_id}")
        return path

    def _is_datatype_exists(self, data_type: str, dataset_id: str) -> bool:
        if self._fs:
            return True
        else:
            return os.path.exists(os.path.join(self._create_path(dataset_id), data_type))

    def _safe_read_from_path(self, data_type: str, dataset_id: str, enforced_schema: Dict[str, Any]) -> pl.DataFrame:
        read_path_str = os.path.join(self._create_path(dataset_id), data_type, "*.parquet")
        read_path = self._fs.glob(read_path_str) if self._fs else glob.glob(read_path_str)
        if self._is_datatype_exists(data_type, dataset_id):
            dataset = pq.ParquetDataset(read_path, filesystem=self._fs)
            try:
                if self._engine == 'pandas':
                    df = dataset.read_pandas().to_pandas()
                elif self._engine == 'polars':
                    df = pl.from_arrow(dataset.read(), schema_overrides=enforced_schema) # TODO: fix schema enforcement on read -- Ram: important for v0.1
                else:
                    raise ValueError("engine must be one of ['pandas', 'polars']")
                return df
            except pl.PanicException as pe:
                print("error, file is not matching Pinecone Datasets Schmea: {}".format(pe), file=sys.stderr)
                raise(pe)
            except Exception as e:
                print("error, no exception: {}".format(e), file=sys.stderr)
                raise(e)
        else:
            print("WARNING: No data found at: {}. Returning empty DF".format(read_path_str), file=sys.stderr)
            if self._engine == 'pandas':
                return pd.DataFrame()
            elif self._engine == 'polars':
                return pl.DataFrame()
            else:
                raise ValueError("engine must be one of ['pandas', 'polars']")

    def _load(self, dataset_id: str) -> None:
        self._documents = self._safe_read_from_path("documents", dataset_id, self._config.Schema.documents)
        self._queries = self._safe_read_from_path("queries", dataset_id, self._config.Schema.queries)

    def __getitem__(self, key: str) -> pl.DataFrame:
        if key in ["documents", "queries"]:
            return getattr(self, key)
        else:
            raise KeyError("Dataset does not have key: {}".format(key))

    @property
    def documents(self) -> pl.DataFrame: 
        return self._documents

    def iter_documents(self, batch_size:int=1) -> Iterator[List[dict[str, Any]]]:
        if isinstance(batch_size, int) and batch_size > 0:
            if self._engine == 'pandas':
                return iter_pandas_dataframe_slices(self._documents[["id", "values", "sparse_values", "metadata"]], batch_size)
            return map(lambda x: x.to_dicts(), self._documents.select(["id", "values", "sparse_values", "metadata"]).iter_slices(n_rows=batch_size))
        else:
            raise ValueError("batch_size must be greater than 0")

    @property
    def queries(self) -> pl.DataFrame:
        return self._queries
    
    def iter_queries(self) -> Iterator[dict[str, Any]]:
        if self._engine == 'pandas':
            return iter_pandas_dataframe_single(self._queries[["values", "sparse_values", "filter", "top_k"]])
        else:
           return self._queries.select(["values", "sparse_values", "filter", "top_k"]).iter_rows(named=True)

    def head(self, n: int = 5) -> pl.DataFrame:
        return self.documents.head(n)
        


