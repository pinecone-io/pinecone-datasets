from pinecone_datasets import load_dataset


class TestLoadDataset:
    def test_load_dataset(self):
        ds = load_dataset("langchain-python-docs-text-embedding-ada-002")
        assert ds is not None

        headdf = ds.head()
        assert headdf is not None
        assert len(headdf) > 0
        columns = headdf.columns.tolist()
        assert "id" in columns
        assert "values" in columns
        assert "sparse_values" in columns
        assert "metadata" in columns
        assert "blob" in columns
