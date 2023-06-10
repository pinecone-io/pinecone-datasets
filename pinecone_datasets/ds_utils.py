from typing import List

import pandas as pd

from pinecone_datasets import Dataset


def transfer_keys_vectorized(df, from_column, to_column, keys):
    # Create new DataFrame to avoid modifying the original one
    df_new = df.copy()

    # Get the dictionaries from the 'from_column' and 'to_column'
    from_dict = df_new[from_column].to_dict()
    to_dict = df_new[to_column].to_dict()

    # Iterate over the dictionaries
    for idx in from_dict.keys():
        # If the 'to_column' dictionary is empty at this index, initialize it
        if not isinstance(to_dict[idx], dict):
            to_dict[idx] = {}

        # Iterate over keys
        for key in keys:
            # Check if the key is in the 'from_column' dictionary
            if key in from_dict[idx]:
                # Move the key from 'from_column' dictionary to 'to_column' dictionary
                to_dict[idx][key] = from_dict[idx].pop(key, None)

    # Update the DataFrame columns with the modified dictionaries
    df_new[from_column] = pd.Series(from_dict)
    df_new[to_column] = pd.Series(to_dict)

    return df_new


def import_documents_keys_from_blob_to_metadata(
    ds: Dataset, keys: List[str]
) -> Dataset:
    """
    Transfers keys from the `blob` column to the `metadata` column in the documents DataFrame.

    Args:
        ds (Dataset): the dataset to modify
        keys (List[str]): the keys to transfer

    Returns:
        Dataset: the modified dataset
    """
    docs = ds.documents

    docs_new = transfer_keys_vectorized(
        docs, from_column="blob", to_column="metadata", keys=keys
    )

    metadata = ds.metadata
    metadata.queries = 0

    return Dataset.from_pandas(
        documents=docs_new, queries=ds.queries, metadata=ds.metadata
    )
