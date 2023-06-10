# Pinecone Datasets

## install

```bash
pip install pinecone-datasets
```

##  Usage - Loading

You can use Pinecone Datasets to load our public datasets or with your own datasets. Datasets library can be used in 2 main ways: ad-hoc loading of datasets from a path or as a catalog loader for datasets. 

### Loading Pinecone Public Datasets (catalog)

Pinecone hosts a public datasets catalog, you can load a dataset by name using `list_datasets` and `load_dataset` functions. This will use the default catalog endpoint (currently GCS) to list and load datasets.

```python
from pinecone_datasets import list_datasets, load_dataset

list_datasets()
# ["quora_all-MiniLM-L6-bm25", ... ]

dataset = load_dataset("quora_all-MiniLM-L6-bm25")

dataset.head()

# Prints
# ┌─────┬───────────────────────────┬─────────────────────────────────────┬───────────────────┬──────┐
# │ id  ┆ values                    ┆ sparse_values                       ┆ metadata          ┆ blob │
# │     ┆                           ┆                                     ┆                   ┆      │
# │ str ┆ list[f32]                 ┆ struct[2]                           ┆ struct[3]         ┆      │
# ╞═════╪═══════════════════════════╪═════════════════════════════════════╪═══════════════════╪══════╡
# │ 0   ┆ [0.118014, -0.069717, ... ┆ {[470065541, 52922727, ... 22364... ┆ {2017,12,"other"} ┆ .... │
# │     ┆ 0.0060...                 ┆                                     ┆                   ┆      │
# └─────┴───────────────────────────┴─────────────────────────────────────┴───────────────────┴──────┘
```

### Expected dataset structure

pinecone datasets can load dataset from every storage where it has access (using the default access: s3, gcs or local permissions)

 we expect data to be uploaded to the following directory structure:

    ├── my-subdir                     # path to where all datasets
    │   ├── my-dataset                # name of dataset
    │   │   ├── metadata.json         # dataset metadata (optional, only for listed)
    │   │   ├── documents             # datasets documents
    │   │   │   ├── file1.parquet      
    │   │   │   └── file2.parquet      
    │   │   ├── queries               # dataset queries
    │   │   │   ├── file1.parquet  
    │   │   │   └── file2.parquet   
    └── ...

The data schema is expected to be as follows:

- `documents` directory contains parquet files with the following schema:
    - Mandatory: `id: str, values: list[float]`
    - Optional: `sparse_values: Dict: indices: List[int], values: List[float]`, `metadata: Dict`, `blob: dict`
        - note: blob is a dict that can contain any data, it is not returned when iterating over the dataset and is inteded to be used for storing additional data that is not part of the dataset schema. however, it is sometime useful to store additional data in the dataset, for example, a document text. In future version this may become a first class citizen in the dataset schema.
- `queries` directory contains parquet files with the following schema:
    - Mandatory: `vector: list[float], top_k: int`
    - Optional: `sparse_vector: Dict: indices: List[int], values: List[float]`, `filter: Dict`
        - note: filter is a dict that contain pinecone filters, for more information see [here](https://docs.pinecone.io/docs/metadata-filtering)

in addition, a metadata file is expected to be in the dataset directory, for example: `s3://my-bucket/my-dataset/metadata.json`

```python
from pinecone_datasets.catalog import DatasetMetadata

meta = DatasetMetadata(
    name="test_dataset",
    created_at="2023-02-17 14:17:01.481785",
    documents=2,
    queries=2,
    source="manual",
    bucket="LOCAL",
    task="unittests",
    dense_model={"name": "bert", "dimension": 3},
    sparse_model={"name": "bm25"},
)
```

full metadata schema can be found in `pinecone_datasets.catalog.DatasetMetadata.schema`

### Loading your own dataset from catalog

To set you own catalog endpoint, set the environment variable `DATASETS_CATALOG_BASEPATH` to your bucket. Note that pinecone uses the default authentication method for the storage type (gcsfs for GCS and s3fs for S3).

```bash
export DATASETS_CATALOG_BASEPATH="s3://my-bucket/my-subdir"
```

```python
from pinecone_datasets import list_datasets, load_dataset

list_datasets()

# ["my-dataset", ... ]

dataset = load_dataset("my-dataset")
```

### Loading your own dataset from path

You can load your own dataset from a local path or a remote path (GCS or S3). Note that pinecone uses the default authentication method for the storage type (gcsfs for GCS and s3fs for S3).

```python
from pinecone_datasets import Dataset

dataset = Dataset("s3://my-bucket/my-subdir/my-dataset")
```

### Loading from a pandas dataframe

Pinecone Datasets enables you to load a dataset from a pandas dataframe. This is useful for loading a dataset from a local file and saving it to a remote storage.
The minimal required data is a documents dataset, and the minimal required columns are `id` and `values`. The `id` column is a unique identifier for the document, and the `values` column is a list of floats representing the document vector.

```python
import pandas as pd

df = pd.read_parquet("my-dataset.parquet")

dataset = Dataset.from_pandas(df)
```

Please check the documentation for more information on the expected dataframe schema. There's also a column mapping variable that can be used to map the dataframe columns to the expected schema.


## Usage - Accessing data

Pinecone Datasets is build on top of pandas. This means that you can use all the pandas API to access the data. In addition, we provide some helper functions to access the data in a more convenient way. 

### Accessing documents and queries dataframes

accessing the documents and queries dataframes is done using the `documents` and `queries` properties. These properties are lazy and will only load the data when accessed. 

```python
document_df: pd.DataFrame = dataset.documents

query_df: pd.DataFrame = dataset.queries
```


## Usage - Iterating

One of the main use cases for Pinecone Datasets is iterating over a dataset. This is useful for upserting a dataset to an index, or for benchmarking. It is also useful for iterating over large datasets - as of today, datasets are not yet lazy, however we are working on it.


```python

# List Iterator, where every list of size N Dicts with ("id", "values", "sparse_values", "metadata")
dataset.iter_documents(batch_size=n) 

# Dict Iterator, where every dict has ("vector", "sparse_vector", "filter", "top_k")
dataset.iter_queries()

```

### The 'blob' column

Pinecone dataset ship with a blob column which is inteneded to be used for storing additional data that is not part of the dataset schema. however, it is sometime useful to store additional data in the dataset, for example, a document text. We added a utility function to move data from the blob column to the metadata column. This is useful for example when upserting a dataset to an index and want to use the metadata to store text data.

```python
from pinecone_datasets import import_documents_keys_from_blob_to_metadata

new_dataset = import_documents_keys_from_blob_to_metadata(dataset, keys=["text"])
```


### upserting to Index

When upserting a Dataset to an Index, only the document data will be upserted to the index. The queries data will be ignored. 

```python
ds = load_dataset("dataset_name")

# If index exists
ds.to_index("index_name")

# If index does not exist use create_index=True, this will create the index with the default pinecone settings and dimension from the dataset metadata.
ds.to_index("index_name", create_index=True)

```

the `to_index` function also accepts additional parameters

* `batch_size` and `concurrency` - for controlling the upserting process
* `kwargs` - for passing additional parameters to the index creation process


## For developers

This project is using poetry for dependency managemet. supported python version are 3.8+. To start developing, on project root directory run:

```bash
poetry install --with dev
```

To run test locally run 

```bash
poetry run pytest --cov pinecone_datasets
```