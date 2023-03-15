__version__ = "0.2.4-alpha"

import warnings

from .dataset import Dataset
from .catalog import Catalog
from . import cfg

catalog = None


def load_dataset(dataset_id: str, **kwargs) -> Dataset:
    """
    Load a dataset from the catalog

    Args:
        dataset_id (str): The name of the dataset to load
        **kwargs: Additional keyword arguments to pass to the Dataset constructor, e.g. `engine='polars'`

    Returns:
        Dataset: A Dataset object

    Examples:
        from pinecone_datasets import load_dataset
        dataset = load_dataset("dataset_name")
    """
    if not catalog:
        raise ValueError(
            "Catalog not initialized. Please call pinecone_datasets.load_catalog() first."
        )
    if dataset_id not in catalog.list_datasets(as_df=False):
        raise FileNotFoundError(f"Dataset {dataset_id} not found in catalog")
    else:
        return Dataset(dataset_id, should_load_metadata=True, **kwargs)


def list_datasets(as_df=False) -> list:
    """
    List all datasets in the catalog, optionally as a pandas DataFrame.
    Catalog is set using the `PINECONE_DATASETS_EDNPOINT` environment variable.
    """
    global catalog
    catalog = Catalog.load()
    return catalog.list_datasets(as_df=as_df)
