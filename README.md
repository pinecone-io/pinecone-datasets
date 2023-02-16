# Pinecone Datasets

## Usage

You can use Pinecone Datasets to load our public datasets or with your own dataset.

### Loading Pinecone Public Datasets

```python
from datasets import list_datasets, load_dataset

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




<!-- ### Loading a dataset from file

```python
dataset = Dataset.from_file("https://storage.googleapis.com/gareth-pinecone-datasets/quora.parquet")

dataset.head()
```

### Loading a dataset from a local directory 

To load data from a local directory we expect data to be uploaded to the following directory structure:

    .
    ├── ...
    ├── path                       # path to where all datasets
    │   ├── dataset_id             # name of dataset
    │   │   ├── documents          # datasets documents
    │   │   │   ├── doc1.parquet  
    │   │   │   └── doc2.parquet   
    │   │   ├── queries            # dataset queries
    │   │   │   ├── q1.parquet  
    │   │   │   └── q2.parquet   
    └── ...
    
Schema for Documents should be 
```python
{
    'id': Utf8,                          # Document ID
    'values': List(Float32),             # Desnse Embeddings
    'sparse_values': Struct([            # Sparse Embeddings
        Field('indices', List(Int32)), 
        Field('values', List(Float32))
    ])
    'metadata': Struct(...)              # String -> Any key value pairs
    'blob': Any                          # Any (document representation)
}
 ```

and for queries
```python
{
    'id': Utf8,                          # Document ID
    'values': List(Float32),             # Desnse Embeddings
    'sparse_values': Struct([            # Sparse Embeddings
        Field('indices', List(Int32)), 
        Field('values', List(Float32))
    ])
    'filter': Struct(...)                # String -> Any key value pairs
    'blob': Any                          # Any (document representation)
}
 ```

```python
from datasets import Dataset

# Dataset(dataset_id: str = None, path: str = None)

dataset = Dataset("two_docs-edo-edo", path="data/")
``` -->

### Iterating over a Dataset documents

```python
# Dict Iterator with ("id", "metadata", "values", "sparse_values")
dataset.iter_docs() 

# List Iterator, where every list of size N Dicts with ("id", "metadata", "values", "sparse_values")
dataset.iter_docs(batch_size=n) 

# polars iter_rows 
# https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/api/polars.DataFrame.iter_rows.html
dataset.documents.iter_rows() 
```

### upserting to Index

```python
# Iterating over documents one by one
for doc in dataset.iter_docs():
    index.upsert(vectors=[doc])

# Iterating over documents in batches
for batch in dataset.iter_docs(batch_size=100):
    index.upsert(vectors=batch)
```

