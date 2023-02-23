# Pinecone Datasets

## Usage

You can use Pinecone Datasets to load our public datasets or with your own dataset.

### Loading Pinecone Public Datasets

```python
from pinecone_datasets import list_datasets, load_dataset

list_datasets()
# ["cc-news_msmarco-MiniLM-L6-cos-v5", ... ]

dataset = load_dataset("cc-news_msmarco-MiniLM-L6-cos-v5")

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
