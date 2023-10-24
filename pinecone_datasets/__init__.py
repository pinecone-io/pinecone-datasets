"""
.. include:: ../README.md
"""

__version__ = "0.6.2"


from .dataset import Dataset, DatasetInitializationError
from .public import list_datasets, load_dataset
from .catalog import DatasetMetadata, DenseModelMetadata
