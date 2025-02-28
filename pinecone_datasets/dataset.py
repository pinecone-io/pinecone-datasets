import logging
from urllib.parse import urlparse
from typing import Any, Generator, Iterator, List, Dict, Optional, Tuple

from .cfg import Schema
from .dataset_metadata import DatasetMetadata
from .fs import get_cloud_fs
from .utils import deprecated

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from .dataset_fsreader import DatasetFSReader
else:
    pd = None  # Placeholder for runtime
    DatasetFSReader = None  # Placeholder for runtime

logger = logging.getLogger(__name__)


def iter_pandas_dataframe_slices(
    df: "pd.DataFrame", batch_size, return_indexes
) -> Generator[List[Dict[str, Any]], None, None]:
    for i in range(0, len(df), batch_size):
        if return_indexes:
            yield (i, df.iloc[i : i + batch_size].to_dict(orient="records"))
        else:
            yield df.iloc[i : i + batch_size].to_dict(orient="records")


def iter_pandas_dataframe_single(
    df: "pd.DataFrame",
) -> Generator[Dict[str, Any], None, None]:
    for i in range(0, len(df), 1):
        yield df.iloc[i : i + 1].to_dict(orient="records")[0]


class Dataset:
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
    def from_pandas(
        cls,
        documents: "pd.DataFrame",
        metadata: DatasetMetadata,
        documents_column_mapping: Optional[Dict] = None,
        queries: Optional["pd.DataFrame"] = None,
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
        instance = cls(dataset_path=None, **kwargs)
        instance._documents = cls._read_pandas_dataframe(
            documents, documents_column_mapping, Schema.Names.documents
        )
        instance._queries = cls._read_pandas_dataframe(
            queries, queries_column_mapping, Schema.Names.queries
        )
        instance._metadata = metadata
        return instance

    @staticmethod
    def _read_pandas_dataframe(
        df: "pd.DataFrame",
        column_mapping: Dict[str, str],
        schema: List[Tuple[str, bool, Any]],
    ) -> "pd.DataFrame":
        """
        Reads a pandas DataFrame and validates it against a schema.

        Args:
            df (pd.DataFrame): the pandas DataFrame to read
            column_mapping (Dict[str, str]): a dictionary mapping the columns of the DataFrame to the Pinecone Datasets Schema (col_name, pinecone_name)
            schema (List[Tuple[str, bool]]): the schema to validate against (column_name, is_nullable)

        Returns:
            pd.DataFrame: the validated, renamed DataFrame
        """
        import pandas as pd

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
                    f"Dataset does not exist at path {self._dataset_path}"
                )
        else:
            self._dataset_path = None
            self._fs = None
        self._documents = None
        self._queries = None
        self._metadata = None

    def __getitem__(self, key: str):
        if key in ["documents", "queries"]:
            return getattr(self, key)
        else:
            raise KeyError("Dataset does not have key: {}".format(key))

    def __len__(self) -> int:
        return self.documents.shape[0]

    @property
    def documents(self) -> "pd.DataFrame":
        if self._documents is None and self._dataset_path is not None:
            from .dataset_fsreader import DatasetFSReader

            self._documents = DatasetFSReader.read_documents(
                self._fs, self._dataset_path
            )
        return self._documents

    @property
    def queries(self) -> "pd.DataFrame":
        if self._queries is None and self._dataset_path is not None:
            from .dataset_fsreader import DatasetFSReader

            self._queries = DatasetFSReader.read_queries(self._fs, self._dataset_path)
        return self._queries

    @property
    def metadata(self) -> DatasetMetadata:
        if self._metadata is None and self._dataset_path is not None:
            from .dataset_fsreader import DatasetFSReader

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

    def head(self, n: int = 5) -> "pd.DataFrame":
        return self.documents.head(n)

    @deprecated
    @classmethod
    def from_catalog(cls, dataset_id, catalog_base_path: str = "", **kwargs):
        """
        DEPRECATED: This method has been removed. Please use `Catalog.load_dataset` instead.
        """
        raise Exception(
            "This method has been removed. Please use `Catalog.load_dataset` instead."
        )

    @deprecated
    def to_catalog(
        self,
        dataset_id: str,
        catalog_base_path: str = "",
        **kwargs,
    ):
        """
        DEPRECATED: This method has been removed. Please use `Catalog.save_dataset` instead.
        """
        raise Exception(
            "This method has been removed. Please use `Catalog.save_dataset` instead."
        )

    @deprecated
    def to_pinecone_index(self, *args, **kwargs):
        """
        DEPRECATED: This method has been removed. Please use the `pinecone.Index.upsert` method instead from the `pinecone` SDK package.
        """
        raise Exception(
            "This method has been removed. Please use the `pinecone.Index.upsert` method instead from the `pinecone` SDK package."
        )
