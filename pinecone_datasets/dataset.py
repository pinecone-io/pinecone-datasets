import sys
import glob
import os
import itertools
import time
import json
import asyncio
import warnings
from urllib.parse import urlparse
from dataclasses import dataclass
from importlib.metadata import version

import gcsfs
import s3fs
import pandas as pd
from tqdm.auto import tqdm
import pyarrow.parquet as pq
from pydantic import ValidationError
from typing import Any, Generator, Iterator, List, Union, Dict, Optional, Tuple

from pinecone_datasets import cfg
from pinecone_datasets.catalog import DatasetMetadata
from pinecone_datasets.fs import get_cloud_fs, LocalFileSystem

if version("pinecone-client").startswith("3"):
    from pinecone import Client as pc, Index
elif version("pinecone-client").startswith("2"):
    import pinecone as pc

    try:
        from pinecone import GRPCIndex as Index
    except ImportError:
        from pinecone import Index
else:
    warnings.warn(
        message="Pinecone client version not supported or non-existent,"
        + "please use pip ineall pinecone-client to install v2 or "
        + "pip install pinecone-datasets[clientv3] to install v3"
    )


class DatasetInitializationError(Exception):
    long_message = """
    This dataset was not initialized from path, but from memory, e.g. Dataset.from_pandas(...)
    Therefore this dataset cannot be reloaded from path, or use methods that require a path.
    If you want to reload a dataset from path, please use the `from_path` method and pass a valid path.
    """

    def __init__(self, message=long_message):
        self.message = message
        super().__init__(self.message)


# TODO: import from Client
@dataclass
class UpsertResponse:
    upserted_count: int


def iter_pandas_dataframe_slices(
    df: pd.DataFrame, batch_size, return_indexes
) -> Generator[List[Dict[str, Any]], None, None]:
    for i in range(0, len(df), batch_size):
        if return_indexes:
            yield (i, df.iloc[i : i + batch_size].to_dict(orient="records"))
        else:
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
        clazz = cls(dataset_path=None, **kwargs)
        clazz._documents = cls._read_pandas_dataframe(
            documents, documents_column_mapping, cfg.Schema.Names.documents
        )
        clazz._queries = cls._read_pandas_dataframe(
            queries, queries_column_mapping, cfg.Schema.Names.queries
        )
        clazz._metadata = metadata
        return clazz

    @staticmethod
    def _read_pandas_dataframe(
        df: pd.DataFrame,
        column_mapping: Dict[str, str],
        schema: List[Tuple[str, bool, Any]],
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
            for column_name, is_nullable, null_value in schema:
                if column_name not in df.columns and not is_nullable:
                    raise ValueError(
                        f"error, file is not matching Pinecone Datasets Schmea: {column_name} not found"
                    )
                elif column_name not in df.columns and is_nullable:
                    df[column_name] = null_value
            return df[[column_name for column_name, _, _ in schema]]

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
        if dataset_path is not None:
            endpoint = urlparse(dataset_path)._replace(path="").geturl()
            self._fs = get_cloud_fs(endpoint, **kwargs)
            self._dataset_path = dataset_path
            if not self._fs.exists(self._dataset_path):
                raise FileNotFoundError(
                    "Dataset does not exist. Please check the path or dataset_id"
                )
        else:
            self._fs = None
            self._dataset_path = None
        self._documents = pd.DataFrame(
            columns=getattr(self._config.Schema.Names, "documents")
        )
        self._queries = pd.DataFrame(
            columns=getattr(self._config.Schema.Names, "queries")
        )
        self._metadata = DatasetMetadata.empty()
        self._pinecone_client = None

    def _is_datatype_exists(self, data_type: str) -> bool:
        if not self._fs:
            raise DatasetInitializationError()
        return self._fs.exists(os.path.join(self._dataset_path, data_type))

    def _safe_read_from_path(self, data_type: str) -> pd.DataFrame:
        if not self._fs:
            raise DatasetInitializationError()

        read_path_str = os.path.join(self._dataset_path, data_type, "*.parquet")
        read_path = self._fs.glob(read_path_str)
        if self._is_datatype_exists(data_type):
            dataset = pq.ParquetDataset(read_path, filesystem=self._fs)
            dataset_schema_names = dataset.schema.names
            columns_to_null = []
            columns_not_null = []
            for column_name, is_nullable, null_value in getattr(
                self._config.Schema.Names, data_type
            ):
                if column_name not in dataset_schema_names and not is_nullable:
                    raise ValueError(
                        f"error, file is not matching Pinecone Datasets Schmea: {column_name} not found"
                    )
                elif column_name not in dataset_schema_names and is_nullable:
                    columns_to_null.append((column_name, null_value))
                else:
                    columns_not_null.append(column_name)
            try:
                # TODO: use of the columns_not_null and columns_to_null is only a workaround for proper schema validation and versioning
                df = dataset.read_pandas(columns=columns_not_null).to_pandas()
                for column_name, null_value in columns_to_null:
                    df[column_name] = null_value
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
        if not self._fs:
            raise DatasetInitializationError()

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

    def __getitem__(self, key: str):
        if key in ["documents", "queries"]:
            return getattr(self, key)
        else:
            raise KeyError("Dataset does not have key: {}".format(key))

    def __len__(self) -> int:
        return self.documents.shape[0]

    @property
    def documents(self) -> pd.DataFrame:
        if self._documents.empty:
            self._documents = self._safe_read_from_path("documents")
        return self._documents

    def iter_documents(
        self, batch_size: int = 1, return_indexes=False
    ) -> Iterator[List[Dict[str, Any]]]:
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
                df=self.documents[self._config.Schema.documents_select_columns].dropna(
                    axis=1, how="all"
                ),
                batch_size=batch_size,
                return_indexes=return_indexes,
            )
        else:
            raise ValueError("batch_size must be greater than 0")

    @property
    def queries(self) -> pd.DataFrame:
        if self._queries.empty:
            self._queries = self._safe_read_from_path("queries")
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
        return iter_pandas_dataframe_single(
            self.queries[self._config.Schema.queries_select_columns]
        )

    @property
    def metadata(self) -> DatasetMetadata:
        if self._metadata.is_empty():
            self._metadata = self._load_metadata()
        return self._metadata

    def head(self, n: int = 5) -> pd.DataFrame:
        return self.documents.head(n)

    def to_path(self, dataset_path: str, **kwargs):
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

    def to_catalog(
        self,
        dataset_id: str,
        catalog_base_path: str = "",
        **kwargs,
    ):
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
        self.to_path(dataset_path, **kwargs)

    async def _async_upsert(self, index_name: str, batch_size: int, concurrency: int):
        pinecone_index = (
            self._pinecone_client.get_index(index_name=index_name)
            if version("pinecone-client").startswith("3")
            else Index(index_name=index_name)
        )

        sem = asyncio.Semaphore(concurrency)

        pinecone_failed_batches = []

        async def send_batch(i, batch):
            async with sem:
                try:
                    return await pinecone_index.upsert(vectors=batch, async_req=True)
                except Exception as pe:
                    if i in pinecone_failed_batches:
                        raise pe
                    else:
                        pinecone_failed_batches.append(i)
                        print(f"failed batches: {pinecone_failed_batches}")
                        return UpsertResponse(upserted_count=0)

        tasks = [
            send_batch(i, chunk)
            for i, chunk in self.iter_documents(
                batch_size=batch_size, return_indexes=True
            )
        ]

        pbar = tqdm(total=len(self.documents), desc="Upserting Vectors")
        total_upserted_count = 0
        for task in asyncio.as_completed(tasks):
            res = await task
            total_upserted_count += res.upserted_count
            pbar.update(res.upserted_count)

        failed_tasks = [
            send_batch(
                i,
                self.documents[self._config.Schema.documents_select_columns]
                .dropna(axis=1, how="all")
                .loc[i : i + batch_size]
                .to_dict(orient="records"),
            )
            for i in pinecone_failed_batches
        ]

        for task in asyncio.as_completed(failed_tasks):
            res = await task
            total_upserted_count += res.upserted_count
            pbar.update(res.upserted_count)

        return {"upserted_count": total_upserted_count}

    def _create_index(
        self,
        index_name: str,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        **kwargs,
    ) -> Index:
        dimension = self.metadata.dense_model.dimension
        api_key = api_key if api_key else os.environ.get("PINECONE_API_KEY", None)
        environment = (
            environment if environment else os.environ.get("PINECONE_ENVIRONMENT", None)
        )

        if not (api_key and environment):
            raise ValueError(
                "Please set PINECONE_API_KEY and PINECONE_ENVIRONMENT environment variables, \
                or pass them as arguments to the function"
            )
        # create client

        if version("pinecone-client").startswith("3"):
            self._pinecone_client = pc(api_key=api_key, region=environment)
        elif version("pinecone-client").startswith("2"):
            pc.init(api_key=api_key, environment=environment)
            self._pinecone_client = pc

        pinecone_index_list = self._pinecone_client.list_indexes()

        if index_name in pinecone_index_list:
            raise ValueError(
                f"index {index_name} already exists, Pinecone Datasets can only be upserted to a new indexe"
            )
        else:
            # create index
            print("creating index")
            try:
                self._pinecone_client.create_index(
                    name=index_name,
                    dimension=self.metadata.dense_model.dimension,
                    **kwargs,
                )
                print("index created")
                return True
            except Exception as e:
                print(f"error creating index: {e}")
                return False

    def to_pinecone_index(
        self,
        index_name: str,
        batch_size: int = 100,
        concurrency: int = 10,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        **kwargs,
    ):
        """
        Saves the dataset to a Pinecone index.

        this function will look for two environment variables:
        - PINECONE_API_KEY
        - PINECONE_ENVIRONMENT

        Then, it will init a Pinecone Client and will perform an upsert to the index.
        The upsert will be using async batches to increase performance.

        Args:
            index_name (str): the name of the index to upsert to
            batch_size (int, optional): the batch size to use for the upsert. Defaults to 100.
            concurrency (int, optional): the concurrency to use for the upsert. Defaults to 10.

        Keyword Args:
            kwargs (Dict): additional arguments to pass to the Pinecone Client constructor when creating the index.


        Returns:
            UpsertResponse: an object containing the upserted_count

        Examples:
            ```python
            result = dataset.to_pinecone_index(index_name="my_index")
            ```
        """

        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError(
                "You are running inside a Jupyter Notebook or another Asyncio context. "
                + "Plesae use the function to_pinecone_index_async instead. "
                + "example: `await dataset.to_pinecone_index_async(index_name)`"
            )

        if not self._create_index(index_name, **kwargs):
            raise RuntimeError("index creation failed")

        cor = self._async_upsert(
            index_name=index_name,
            batch_size=batch_size,
            concurrency=concurrency,
        )
        return asyncio.run(cor)

    async def to_pinecone_index_async(
        self,
        index_name: str,
        batch_size: int = 100,
        concurrency: int = 10,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        **kwargs,
    ):
        """
        Saves the dataset to a Pinecone index.

        this function will look for two environment variables:
        - PINECONE_API_KEY
        - PINECONE_ENVIRONMENT

        Then, it will init a Pinecone Client and will perform an upsert to the index.
        The upsert will be using async batches to increase performance.

        Args:
            index_name (str): the name of the index to upsert to
            batch_size (int, optional): the batch size to use for the upsert. Defaults to 100.
            concurrency (int, optional): the concurrency to use for the upsert. Defaults to 10.

        Keyword Args:
            kwargs (Dict): additional arguments to pass to the Pinecone Client constructor when creating the index.

        Returns:
            UpsertResponse: an object containing the upserted_count

        Examples:
            ```python
            result = await dataset.to_pinecone_index_async(index_name="my_index")
            ```
        """
        loop = asyncio.get_event_loop()
        if not loop.is_running():
            raise RuntimeError(
                "You are running inside a Jupyter Notebook or another Asyncio context. \
                Plesae use the function to_pinecone_index instead. \
                example: `dataset.to_pinecone_index(index_name)`"
            )

        if not self._create_index(index_name, **kwargs):
            raise RuntimeError("index creation failed")

        res = await self._async_upsert(
            index_name=index_name,
            batch_size=batch_size,
            concurrency=concurrency,
        )

        return res
