__version__ = '0.1.0'

from .dataset import Dataset
from .catalog import Catalog
from . import cfg

catalog = Catalog.load()

def load_dataset(dataset_id: str) -> Dataset:
    return Dataset(dataset_id)

def list_datasets() -> list:
    return catalog.list_datasets()