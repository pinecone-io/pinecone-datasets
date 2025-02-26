from pinecone_datasets import list_datasets
class TestListDatasets:
    def test_list_datasets(self):
        datasets = list_datasets()
        assert len(datasets) > 0
        assert "quora_all-MiniLM-L6-bm25" in datasets
