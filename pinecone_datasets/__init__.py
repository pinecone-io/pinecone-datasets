"""
.. include:: ../README.md
"""

__version__ = "1.0.2"


from .cache import cache_info as cache_info
from .cache import clear_cache as clear_cache
from .cache import set_cache_dir as set_cache_dir
from .catalog import Catalog as Catalog
from .dataset import Dataset as Dataset
from .dataset_metadata import DatasetMetadata as DatasetMetadata
from .dataset_metadata import DenseModelMetadata as DenseModelMetadata
from .public import list_datasets as list_datasets
from .public import load_dataset as load_dataset
