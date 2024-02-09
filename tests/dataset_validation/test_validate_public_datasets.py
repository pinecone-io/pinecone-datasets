import pytest
import pinecone_datasets


def pytest_generate_tests(metafunc):
    # Discover the set of datasets in the public repo, populating the
    # 'dataset' parameter with them all.
    metafunc.parametrize("dataset", pinecone_datasets.list_datasets())


def test_all_datasets_valid(dataset):
    """For the given dataset, check we can successfully load it from cloud
    storage (i.e. metadata checks pass and necessary files are present"""
    ds = pinecone_datasets.load_dataset(dataset)
    # Ideally should check all sets for this, but some are _very_ big and OOM kill
    # a typical VM
    if ds.metadata.documents > 2_000_000:
        pytest.skip(
            f"Skipping dataset '{dataset} which is larger than 2,000,000 vectors (has {ds.metadata.documents:,})"
        )
    df = ds.documents
    duplicates = df[df["id"].duplicated()]
    num_duplicates = len(duplicates)
    if num_duplicates:
        print("Summary of duplicate IDs in vectors:")
        print(duplicates)
    assert (
        num_duplicates == 0
    ), f"Not all vector ids are unique - found {len(duplicates)} duplicates out of {len(df)} total vectors"

    assert ds.metadata.documents == len(
        df
    ), f"Count of vectors found in Dataset file ({len(ds.documents)}) does not match count in metadata ({ds.metadata.documents})"
