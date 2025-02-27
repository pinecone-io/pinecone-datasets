import logging
import os
from urllib.parse import urlparse
from dataclasses import dataclass


import pandas as pd
from typing import Any, Generator, Iterator, List, Dict, Optional, Tuple

from .cfg import Schema, Storage
from .dataset_metadata import DatasetMetadata
from .fs import get_cloud_fs
from .utils import deprecated
from .dataset_fswriter import DatasetFSWriter
from .dataset_fsreader import DatasetFSReader
from .index_writer import IndexWriter

logger = logging.getLogger(__name__)


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
            else os.environ.get("DATASETS_CATALOG_BASEPATH", Storage.endpoint)
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
            documents, documents_column_mapping, Schema.Names.documents
        )
        clazz._queries = cls._read_pandas_dataframe(
            queries, queries_column_mapping, Schema.Names.queries
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
            return pd.DataFrame(columns=[column_name for column_name, _, _ in schema])
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
        self._documents = None
        self._queries = None
        self._metadata = None
        self._pinecone_client = None

    @deprecated
    def to_catalog(
        self,
        dataset_id: str,
        catalog_base_path: str = "",
        **kwargs,
    ):
        """
        Saves the dataset to the public catalog.
        """
        raise Exception(
            "This method is deprecated. Please use `Catalog.save_dataset` instead."
        )

    def _is_datatype_exists(self, data_type: str) -> bool:
        if not self._fs:
            raise DatasetInitializationError()
        return self._fs.exists(os.path.join(self._dataset_path, data_type))

    def __getitem__(self, key: str):
        if key in ["documents", "queries"]:
            return getattr(self, key)
        else:
            raise KeyError("Dataset does not have key: {}".format(key))

    def __len__(self) -> int:
        return self.documents.shape[0]

    @property
    def documents(self) -> pd.DataFrame:
        if self._documents is None:
            if not self._fs:
                raise DatasetInitializationError()
            self._documents = DatasetFSReader.read_documents(
                self._fs, self._dataset_path
            )
        return self._documents

    @property
    def queries(self) -> pd.DataFrame:
        if self._queries is None:
            if not self._fs:
                raise DatasetInitializationError()
            self._queries = DatasetFSReader.read_queries(self._fs, self._dataset_path)
        return self._queries

    @property
    def metadata(self) -> DatasetMetadata:
        if self._metadata is None:
            if not self._fs:
                raise DatasetInitializationError()
            self._metadata = DatasetFSReader.read_metadata(self._fs, self._dataset_path)
        return self._metadata

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
                df=self.documents[Schema.documents_select_columns].dropna(
                    axis=1, how="all"
                ),
                batch_size=batch_size,
                return_indexes=return_indexes,
            )
        else:
            raise ValueError("batch_size must be greater than 0")

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
        return iter_pandas_dataframe_single(self.queries[Schema.queries_select_columns])

    def head(self, n: int = 5) -> pd.DataFrame:
        return self.documents.head(n)

    def to_path(self, dataset_path: str, **kwargs):
        """
        Saves the dataset to a local or cloud storage path.
        """
        DatasetFSWriter.write_dataset(dataset_path, self, **kwargs)

    @deprecated
    def to_pinecone_index(
        self,
        index_name: str,
        namespace: Optional[str] = "",
        should_create_index: bool = True,
        batch_size: int = 100,
        show_progress: bool = True,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        region: Optional[str] = None,
        cloud: Optional[str] = None,
        serverless: Optional[bool] = None,
        **kwargs,
    ):
        """
        Saves the dataset to a Pinecone index.

        this function will look for four environment variables:
        - SERVERLESS
        - PINECONE_API_KEY
        - PINECONE_REGION
        - PINECONE_CLOUD
        - PINECONE_ENVIRONMENT

        Then, it will init a Pinecone Client and will perform an upsert to the index.
        The upsert will be using async batches to increase performance.

        Args:
            index_name (str): the name of the index to upsert to
            api_key (str, optional): the api key to use for the upsert. Defaults to None.
            region (str, optional): the region to use for the upsert for serverless. Defaults to None.
            cloud (str, optional): the cloud to use for the upsert for serverless. Defaults to None.
            environment (str, optional): the environment to use for the upsert for pod-based. Defaults to None.
            serverless (bool, optional): whether to use serverless or pod-based. Defaults to None.
            namespace (str, optional): the namespace to use for the upsert. Defaults to "".
            batch_size (int, optional): the batch size to use for the upsert. Defaults to 100.
            show_progress (bool, optional): whether to show a progress bar while upserting. Defaults to True.

        Keyword Args:
            kwargs (Dict): additional arguments to pass to the Pinecone Client constructor when creating the index.
            see available parameters here: https://docs.pinecone.io/reference/create_index


        Returns:
            UpsertResponse: an object containing the upserted_count

        Examples:
            ```python
            result = dataset.to_pinecone_index(index_name="my_index")
            ```
        """
        index_writer = IndexWriter(api_key=api_key, **kwargs)

        if should_create_index:
            if environment is not None and (cloud is not None or region is not None):
                raise ValueError(
                    "environment, cloud, and region should not all be provided; environment is used with pod-based indexes while cloud and region are used with serverless indexes"
                )

            is_serverless = (
                serverless
                or os.environ.get("SERVERLESS", False)
                or (cloud is not None and region is not None)
            )
            if is_serverless:
                index_writer.create_serverless_index(
                    index_name=index_name,
                    dimension=self.metadata.dense_model.dimension,
                    cloud=cloud or os.getenv("PINECONE_CLOUD", "aws"),
                    region=region or os.getenv("PINECONE_REGION", "us-west2"),
                )
            else:
                index_writer.create_pod_index(
                    index_name=index_name,
                    dimension=self.metadata.dense_model.dimension,
                    environment=environment or os.environ["PINECONE_ENVIRONMENT"],
                )

        return index_writer.upsert_to_index(
            index_name=index_name,
            df=self.documents,
            namespace=namespace,
            batch_size=batch_size,
            show_progress=show_progress,
        )
