import json

import gcsfs
import polars as pl

class Catalog(object):
    def __init__(self) -> None:
        gcs_file_system = gcsfs.GCSFileSystem(token='anon')
        gcs_json_path = "gs://pinecone-datasets-dev/catalog.json"
        with gcs_file_system.open(gcs_json_path) as f:
            self._catalog = pl.from_dicts(json.load(f))

    def is_in_catalog(self, dataset_id: str) -> bool:
        filtered_catalog = self._catalog.filter(pl.col("name") == dataset_id)
        if filtered_catalog.shape[0] == 0:
            return False
        elif filtered_catalog.shape[0] == 1:
            return True
        else:
            raise ValueError("There is more than one dataset with the same name")
        
    def get_dataset(self, dataset_id: str) -> pl.DataFrame:
        if self.is_in_catalog(dataset_id):
            return self._catalog.filter(pl.col("name") == dataset_id).to_dict()
        else:
            raise ValueError("Dataset not found in catalog")

    def list_datasets(self) -> list:
        return self._catalog["name"].to_list()