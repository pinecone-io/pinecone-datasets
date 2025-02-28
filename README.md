# Pinecone Datasets

## install

```bash
pip install pinecone-datasets
```

### Loading public datasets

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


## Usage - Accessing data

Each dataset has three main attributes, `documents`, `queries`, and `metadata` which are lazily loaded the first time they are accessed. You may notice a delay as the underlying parquet files are being downloaded the first time these attributes are accessed.

Pinecone Datasets is build on top of pandas. `documents` and `queries` are lazily-loaded pandas dataframes. This means that you can use all the pandas API to access the data. In addition, we provide some helper functions to access the data in a more convenient way. 

accessing the documents and queries dataframes is done using the `documents` and `queries` properties. These properties are lazy and will only load the data when accessed. 

```python
from pinecone_datasets import list_datasets, load_dataset

dataset = load_dataset("quora_all-MiniLM-L6-bm25")

document_df: pd.DataFrame = dataset.documents

query_df: pd.DataFrame = dataset.queries
```


## Usage - Iterating over documents

The `Dataset` class has helpers for iterating over your dataset. This is useful for upserting a dataset to an index, or for benchmarking.

```python

# List Iterator, where every list of size N Dicts with ("id", "values", "sparse_values", "metadata")
dataset.iter_documents(batch_size=n) 

# Dict Iterator, where every dict has ("vector", "sparse_vector", "filter", "top_k")
dataset.iter_queries()
```

### Upserting to Index

To upsert data to the index, you should install the [Pinecone SDK](https://github.com/pinecone-io/pinecone-python-client)

```python
from pinecone import Pinecone, ServerlessSpec
from pinecone_datasets import load_dataset, list_datasets

# See what datasets are available
for ds in list_datasets():
    print(ds)

# Download embeddings data 
dataset = load_dataset(dataset_name)

# Instantiate a Pinecone client using API key from app.pinecone.io
pc = Pinecone(api_key='key')

# Create a Pinecone index
index_config = pc.create_index(
    name="demo-index",
    dimension=dataset.metadata.dense_model.dimension,
    spec=ServerlessSpec(cloud="aws", region="us-east1")
)

# Instantiate an index client
index = pc.Index(host=index_config.host)

# Upsert data from the dataset
index.upsert_from_dataframe(df=dataset.documents)
```
