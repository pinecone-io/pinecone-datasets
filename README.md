# Pinecone Datasets

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
 ┌─────┬───────────────────────────┬─────────────────────────────────────┬───────────────────┬──────┐
 │ id  ┆ values                    ┆ sparse_values                       ┆ metadata          ┆ blob │
 │ --- ┆ ---                       ┆ ---                                 ┆ ---               ┆ ---  │
 │ str ┆ list[f32]                 ┆ struct[2]                           ┆ struct[3]         ┆      │
 ╞═════╪═══════════════════════════╪═════════════════════════════════════╪═══════════════════╪══════╡
 │ 0   ┆ [0.118014, -0.069717, ... ┆ {[470065541, 52922727, ... 22364... ┆ {2017,12,"other"} ┆ .... │
 │     ┆ 0.0060...                 ┆                                     ┆                   ┆      │
 └─────┴───────────────────────────┴─────────────────────────────────────┴───────────────────┴──────┘
```


### Iterating over a Dataset documents

```python

# List Iterator, where every list of size N Dicts with ("id", "metadata", "values", "sparse_values")
dataset.iter_documents(batch_size=n) 
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

# Or: Iterating over documents in batches
for batch in dataset.iter_documents(batch_size=100):
    index.upsert(vectors=batch)
```

#### upserting to an index with GRPC

Simply use GRPCIndex and do:

```python
index = pinecone.GRPCIndex("my-index")

# Iterating over documents in batches
for batch in dataset.iter_documents(batch_size=100):
    index.upsert(vectors=batch)
```

## For developers

This project is using poetry for dependency managemet. supported python version are 3.8+. To start developing, on project root directory run:

```bash
poetry install
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
    created_at="2021-09-01",
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
DatasetMetadata.schema()
```
