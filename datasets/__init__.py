__version__ = '0.1.0'

from .dataset import Dataset
from .catalog import Catalog
from . import cfg

catalog = Catalog.load()

def load_dataset(dataset_id: str, **kwargs) -> Dataset:
    return Dataset(dataset_id, **kwargs)

def list_datasets() -> list:
    return catalog.list_datasets()