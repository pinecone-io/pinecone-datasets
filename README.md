# Pinecone Datasets

## install

```bash
pip install pinecone-datasets
```

## Usage

You can use Pinecone Datasets to load our public datasets or with your own dataset.

### Loading Pinecone Public Datasets

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


### Iterating over a Dataset documents and queries

Iterating over documents is useful for upserting but also for different updating. Iterating over queries is helpful in benchmarking

```python

# List Iterator, where every list of size N Dicts with ("id", "metadata", "values", "sparse_values")
dataset.iter_documents(batch_size=n) 

dataset.iter_queries()

```

### upserting to Index

```bash
pip install pinecone-client
```

```python
import pinecone
pinecone.init(api_key="API_KEY", environment="us-west1-gcp")

pinecone.create_index(name="my-index", dimension=384, pod_type='s1')

index = pinecone.Index("my-index")

# you can iterate over documents in batches
for batch in dataset.iter_documents(batch_size=100):
    index.upsert(vectors=batch)

# or upsert the dataset as dataframe
index.upsert_from_dataframe(dataset.drop(columns=["blob"]))

# using gRPC
index = pinecone.GRPCIndex("my-index")
```

## Advanced Usage

### Working with your own dataset storage

Datasets is using Pinecone's public datasets bucket on GCS, you can use your own bucket by setting the `DATASETS_CATALOG_BASEPATH` environment variable.

```bash
export PINECONE_DATASETS_ENDPOINT="gs://my-bucket"
```

this will change the default endpoint to your bucket, and upon calling `list_datasets` or `load_dataset` it will scan your bucket and list all datasets.

Note that you can also use `s3://` as a prefix to your bucket.

### Authenication to your own bucket

For now, Pinecone Datastes supports only GCS and S3 buckets, and with default authentication as provided by the fsspec implementation, respectively: `gcsfs` and `s3fs`.

### Using aws key/secret authentication methods

first, to set a new endpoint, set the environment variable `PINECONE_DATASETS_ENDPOINT` to your bucket.

```bash
export PINECONE_DATASETS_ENDPOINT="s3://my-bucket"
```

then, you can use the `key` and `secret` parameters to pass your credentials to the `list_datasets` and `load_dataset` functions.

```python
st = list_datasets(
        key=os.environ.get("S3_ACCESS_KEY"),
        secret=os.environ.get("S3_SECRET"),
    )

ds = load_dataset(
        "test_dataset",
        key=os.environ.get("S3_ACCESS_KEY"),
        secret=os.environ.get("S3_SECRET"),
)
```

## For developers

This project is using poetry for dependency managemet. supported python version are 3.8+. To start developing, on project root directory run:

```bash
poetry install --with dev
```

To run test locally run 

```bash
poetry run pytest --cov pinecone_datasets
```

To create a pinecone-public dataset you may need to generate a dataset metadata. For example:

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

to see the complete schema you can run:

```python
meta.schema()
```

in order to list a dataset you can save dataset metadata (NOTE: write permission to loacaion is needed)

```python
dataset = Dataset("non-listed-dataset")
dataset._save_metadata(meta)
```

### Uploading and listing a dataset. 

pinecone datasets can load dataset from every storage where it has access (using the default access: s3, gcs or local permissions)

 we expect data to be uploaded to the following directory structure:

    ├── base_path                     # path to where all datasets
    │   ├── dataset_id                # name of dataset
    │   │   ├── metadata.json         # dataset metadata (optional, only for listed)
    │   │   ├── documents             # datasets documents
    │   │   │   ├── file1.parquet      
    │   │   │   └── file2.parquet      
    │   │   ├── queries               # dataset queries
    │   │   │   ├── file1.parquet  
    │   │   │   └── file2.parquet   
    └── ...

a listed dataset is a dataset that is loaded and listed using `load_dataset` and `list_dataset`
pinecone datasets will scan storage and will list every dataset with metadata file, for example: `s3://my-bucket/my-dataset/metadata.json`

### Accessing a non-listed dataset

to access a non listed dataset you can directly load it via:

```python
from pinecone_datasets import Dataset

dataset = Dataset("non-listed-dataset")
```


