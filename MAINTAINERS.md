# Pinecone Datasets

### Supported storage options

pinecone_datasets can load datasets from Google Cloud storage, Amazon S3, and local files.

By default, the `load_dataset` and `list_datasets` packages will pull from Pinecone's public GCS bucket at `gs://pinecone-datasets-dev`, but you can interact with catalogs stored in other locations.

```python
from pinecone_datasets import Catalog

# Local catalog
catalog = Catalog(base_path="/path/to/local/catalog")
catalog.list_datasets()

# Google Cloud
catalog = Catalog(base_path="gs://bucket-name")

# S3 catalog
s3_catalog = Catalog(base_path="s3://bucket-name")
```

If you are using Amazon S3 or Google Cloud to access private buckets, you can use environment variables to configure your credentials. For example, if you set a base_path starting with "gs://", the `gcsfs` package will attempt to find credentials by looking in cache locations used by `gcloud auth login` or reading environment variables such as `GOOGLE_APPLICATION_CREDENTIALS`.

## Adding a new dataset to the public datasets repo

Note: Only Pinecone employees with access to the bucket can complete this step.

Prerequisites:

1. Install google cloud CLI
2. Authenticate with `gcloud auth login`

```python
from pinecone_datasets import Catalog, Dataset, DatasetMetadata, DenseModelMetadata

# 1. Prepare pandas dataframes containing your embeddings
documents_df = ...
queries_df = ...

# 2. Create metadata to describe the dataset
import datatime
metadata = DatasetMetadata(
    name="new-dataset-name",
    created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
    documents=len(documents_df),
    queries=len(queries_df),
    dense_model=DenseModelMetadata(
        name="ada2",
        dimension=2,
    ),
)

# 3. Take all this, and instantiate a Dataset
ds = Dataset.from_pandas(
    documents=documents_df,
    queries=queries_df,
    metadata=metadata
)

# 4. Save to catalog (requires gcloud auth step above)
catalog = Catalog(base_path="gs://pinecone-datasets-dev")
catalog.save_dataset(ds)
```

Afterwards, verify the new dataset appears in list function and can be used

```python
from pinecone_datasets import list_datasets, load_dataset

list_datasets(as_df=True)

ds = load_dataset("new-dataset-name")
ds.documents
ds.head()
```

### Expected dataset structure

The package expects data to be laid out with the following directory structure:

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

full metadata schema can be found in `pinecone_datasets.dataset_metadata.DatasetMetadata.schema`

### The 'blob' column

Pinecone dataset ship with a blob column which is inteneded to be used for storing additional data that is not part of the dataset schema. however, it is sometime useful to store additional data in the dataset, for example, a document text. We added a utility function to move data from the blob column to the metadata column. This is useful for example when upserting a dataset to an index and want to use the metadata to store text data.

```python
from pinecone_datasets import import_documents_keys_from_blob_to_metadata

new_dataset = import_documents_keys_from_blob_to_metadata(dataset, keys=["text"])
```

## Usage saving

You can save your dataset to a catalog managed by you or to a local path or a remote path (GCS or S3). 

### Saving a dataset to a Catalog

To set you own catalog endpoint, set the environment variable `DATASETS_CATALOG_BASEPATH` to your bucket. Note that pinecone uses the default authentication method for the storage type (gcsfs for GCS and s3fs for S3).

After this environment variable is set you can save your dataset to the catalog using the `save` function

```python
from pinecone_datasets import Dataset

metadata = DatasetMetadata(**{"name": "my-dataset", ...})
```


### Saving to Path

You can save your dataset to a local path or a remote path (GCS or S3). Note that pinecone uses the default authentication method for the storage type (gcsfs for GCS and s3fs for S3).

```python
dataset = Dataset.from_pandas(documents, queries, metadata)
dataset.to_path("s3://my-bucket/my-subdir/my-dataset")
```

## Running tests

This project is using poetry for dependency managemet. To start developing, on project root directory run:

```bash
poetry install --with dev
```

To run test locally run 

```bash
poetry run pytest test/unit --cov pinecone_datasets
```

## Code formatting and linting

This project uses [ruff](https://github.com/astral-sh/ruff) for code formatting and linting. Ruff is a fast, modern Python linter and formatter written in Rust.

To format code:

```bash
poetry run ruff format .
```

To check code formatting without making changes:

```bash
poetry run ruff format --check .
```

To run linting checks:

```bash
poetry run ruff check .
```

To automatically fix linting issues:

```bash
poetry run ruff check --fix .
```
