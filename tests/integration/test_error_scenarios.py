import concurrent.futures
import json
import os

import pandas as pd
import pytest
from unittest.mock import patch

from pinecone_datasets import Catalog, Dataset, DatasetMetadata, DenseModelMetadata
from pinecone_datasets.dataset_fsreader import DatasetFSReader
from pinecone_datasets.dataset_fswriter import DatasetFSWriter


class TestIntegrationErrorScenarios:
    """Integration tests for complex error scenarios"""

    def test_concurrent_read_operations(self, tmpdir):
        """Test concurrent read operations on same dataset"""
        # Create a valid dataset
        dataset_path = str(tmpdir.mkdir("dataset"))
        documents_path = os.path.join(dataset_path, "documents")
        os.makedirs(documents_path)

        documents_data = pd.DataFrame(
            [
                {
                    "id": str(i),
                    "values": [float(i), float(i + 1)],
                    "metadata": {"index": i},
                }
                for i in range(100)
            ]
        )
        documents_data.to_parquet(os.path.join(documents_path, "part-0.parquet"))

        metadata = {
            "name": "test",
            "created_at": "2021-01-01 00:00:00.000000",
            "documents": 100,
            "queries": 0,
            "dense_model": {"name": "ada2", "dimension": 2},
        }
        with open(os.path.join(dataset_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        # Perform concurrent reads
        def read_dataset():
            ds = Dataset.from_path(dataset_path)
            return len(ds.documents)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(read_dataset) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All reads should succeed
        assert all(r == 100 for r in results)

    def test_concurrent_write_operations(self, tmpdir):
        """Test concurrent write operations to different datasets"""
        catalog_path = str(tmpdir.mkdir("catalog"))
        catalog = Catalog(base_path=catalog_path)

        def write_dataset(index):
            documents = pd.DataFrame(
                [
                    {
                        "id": f"{index}-{i}",
                        "values": [float(i), float(i + 1)],
                        "metadata": {"index": i},
                    }
                    for i in range(10)
                ]
            )

            metadata = DatasetMetadata(
                name=f"dataset_{index}",
                created_at="2021-01-01 00:00:00.000000",
                documents=10,
                queries=0,
                dense_model=DenseModelMetadata(name="ada2", dimension=2),
            )

            dataset = Dataset.from_pandas(
                documents=documents, queries=None, metadata=metadata
            )
            catalog.save_dataset(dataset)
            return index

        # Write multiple datasets concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(write_dataset, i) for i in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All writes should succeed
        assert len(results) == 5
        assert set(results) == {0, 1, 2, 3, 4}

        # Verify all datasets were created
        for i in range(5):
            dataset_path = os.path.join(catalog_path, f"dataset_{i}")
            assert os.path.exists(dataset_path)

    def test_concurrent_read_write_same_location(self, tmpdir):
        """Test concurrent read and write to same location causes appropriate errors"""
        dataset_path = str(tmpdir.mkdir("dataset"))
        documents_path = os.path.join(dataset_path, "documents")
        os.makedirs(documents_path)

        # Create initial dataset
        documents_data = pd.DataFrame(
            [{"id": "1", "values": [0.1, 0.2], "metadata": {"key": "value"}}]
        )
        documents_data.to_parquet(os.path.join(documents_path, "part-0.parquet"))

        metadata = {
            "name": "test",
            "created_at": "2021-01-01 00:00:00.000000",
            "documents": 1,
            "queries": 0,
            "dense_model": {"name": "ada2", "dimension": 2},
        }
        with open(os.path.join(dataset_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        errors = []

        def read_dataset():
            try:
                ds = Dataset.from_path(dataset_path)
                return len(ds.documents)
            except Exception as e:
                errors.append(e)
                return None

        def write_dataset():
            try:
                documents = pd.DataFrame(
                    [{"id": "2", "values": [0.3, 0.4], "metadata": {"key": "value2"}}]
                )

                meta = DatasetMetadata(
                    name="test",
                    created_at="2021-01-01 00:00:00.000000",
                    documents=1,
                    queries=0,
                    dense_model=DenseModelMetadata(name="ada2", dimension=2),
                )

                ds = Dataset.from_pandas(
                    documents=documents, queries=None, metadata=meta
                )
                DatasetFSWriter.write_dataset(dataset_path, ds)
                return True
            except Exception as e:
                errors.append(e)
                return False

        # Perform concurrent reads and writes
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            futures.extend([executor.submit(read_dataset) for _ in range(2)])
            futures.extend([executor.submit(write_dataset) for _ in range(2)])
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Some operations should succeed, and we shouldn't have crashes
        assert len(results) == 4

    def test_large_dataset_memory_handling(self, tmpdir):
        """Test handling of large dataset"""
        dataset_path = str(tmpdir.mkdir("dataset"))
        documents_path = os.path.join(dataset_path, "documents")
        os.makedirs(documents_path)

        # Create a moderately large dataset
        num_rows = 10000
        documents_data = pd.DataFrame(
            [
                {
                    "id": str(i),
                    "values": [float(i) for _ in range(100)],  # 100-dimensional vectors
                    "metadata": {"index": i, "description": f"document_{i}" * 10},
                }
                for i in range(num_rows)
            ]
        )

        documents_data.to_parquet(os.path.join(documents_path, "part-0.parquet"))

        metadata = {
            "name": "large_dataset",
            "created_at": "2021-01-01 00:00:00.000000",
            "documents": num_rows,
            "queries": 0,
            "dense_model": {"name": "ada2", "dimension": 100},
        }
        with open(os.path.join(dataset_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        # Load and verify
        ds = Dataset.from_path(dataset_path)
        assert len(ds.documents) == num_rows

        # Test iteration (more memory efficient than loading all at once)
        count = 0
        for batch in ds.iter_documents(batch_size=100):
            count += len(batch)
        assert count == num_rows

    def test_multiple_parquet_files_with_errors(self, tmpdir):
        """Test loading dataset with multiple parquet files, some with errors"""
        dataset_path = str(tmpdir.mkdir("dataset"))
        documents_path = os.path.join(dataset_path, "documents")
        os.makedirs(documents_path)

        # Create valid parquet files
        for i in range(3):
            data = pd.DataFrame(
                [
                    {
                        "id": f"{i}-{j}",
                        "values": [float(j), float(j + 1)],
                        "metadata": {"index": j},
                    }
                    for j in range(10)
                ]
            )
            data.to_parquet(os.path.join(documents_path, f"part-{i}.parquet"))

        # Add a corrupted parquet file
        with open(os.path.join(documents_path, "part-3.parquet"), "w") as f:
            f.write("corrupted data")

        metadata = {
            "name": "test",
            "created_at": "2021-01-01 00:00:00.000000",
            "documents": 30,
            "queries": 0,
            "dense_model": {"name": "ada2", "dimension": 2},
        }
        with open(os.path.join(dataset_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        # Should fail when encountering corrupted file
        from fsspec.implementations.local import LocalFileSystem

        fs = LocalFileSystem()

        with pytest.raises(Exception):
            DatasetFSReader.read_documents(fs, dataset_path)

    def test_partial_dataset_write(self, tmpdir):
        """Test recovery from partial dataset write"""
        catalog_path = str(tmpdir.mkdir("catalog"))
        catalog = Catalog(base_path=catalog_path)

        documents = pd.DataFrame(
            [{"id": "1", "values": [0.1, 0.2], "metadata": {"key": "value"}}]
        )

        metadata = DatasetMetadata(
            name="partial_dataset",
            created_at="2021-01-01 00:00:00.000000",
            documents=1,
            queries=0,
            dense_model=DenseModelMetadata(name="ada2", dimension=2),
        )

        dataset = Dataset.from_pandas(
            documents=documents, queries=None, metadata=metadata
        )

        # Mock to simulate failure during metadata write
        def failing_write_metadata(fs, dataset_path, dataset):
            # Documents already written, now fail on metadata
            raise OSError("Simulated failure during metadata write")

        with patch.object(
            DatasetFSWriter, "_write_metadata", side_effect=failing_write_metadata
        ):
            with pytest.raises(IOError):
                catalog.save_dataset(dataset)

        # Dataset directory should exist but be incomplete
        dataset_path = os.path.join(catalog_path, "partial_dataset")
        if os.path.exists(dataset_path):
            # If documents directory exists, metadata.json should not
            documents_dir = os.path.join(dataset_path, "documents")
            metadata_file = os.path.join(dataset_path, "metadata.json")
            if os.path.exists(documents_dir):
                assert not os.path.exists(metadata_file)

    def test_empty_dataset_handling(self, tmpdir):
        """Test handling of completely empty dataset"""
        dataset_path = str(tmpdir.mkdir("dataset"))

        # Create directories but no data
        os.makedirs(os.path.join(dataset_path, "documents"))

        metadata = {
            "name": "empty_dataset",
            "created_at": "2021-01-01 00:00:00.000000",
            "documents": 0,
            "queries": 0,
            "dense_model": {"name": "ada2", "dimension": 2},
        }
        with open(os.path.join(dataset_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        from fsspec.implementations.local import LocalFileSystem

        fs = LocalFileSystem()

        # Should raise ValueError when no parquet files found in existing directory
        with pytest.raises(ValueError, match="No parquet files found"):
            DatasetFSReader.read_documents(fs, dataset_path)

    def test_dataset_with_missing_optional_fields(self, tmpdir):
        """Test dataset where all optional fields are missing"""
        dataset_path = str(tmpdir.mkdir("dataset"))
        documents_path = os.path.join(dataset_path, "documents")
        os.makedirs(documents_path)

        # Documents with only required fields
        documents_data = pd.DataFrame([{"id": "1", "values": [0.1, 0.2]}])
        documents_data.to_parquet(os.path.join(documents_path, "part-0.parquet"))

        metadata = {
            "name": "minimal_dataset",
            "created_at": "2021-01-01 00:00:00.000000",
            "documents": 1,
            "queries": 0,
            "dense_model": {"name": "ada2", "dimension": 2},
        }
        with open(os.path.join(dataset_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        # Should load successfully and add None for optional fields
        ds = Dataset.from_path(dataset_path)
        assert len(ds.documents) == 1
        assert "metadata" in ds.documents.columns
        assert "sparse_values" in ds.documents.columns

    def test_catalog_with_many_datasets(self, tmpdir):
        """Test catalog operations with many datasets"""
        catalog_path = str(tmpdir.mkdir("catalog"))
        catalog = Catalog(base_path=catalog_path)

        # Create 50 datasets
        num_datasets = 50
        for i in range(num_datasets):
            documents = pd.DataFrame(
                [
                    {
                        "id": "1",
                        "values": [float(i), float(i + 1)],
                        "metadata": {"index": i},
                    }
                ]
            )

            metadata = DatasetMetadata(
                name=f"dataset_{i:03d}",
                created_at="2021-01-01 00:00:00.000000",
                documents=1,
                queries=0,
                dense_model=DenseModelMetadata(name="ada2", dimension=2),
            )

            dataset = Dataset.from_pandas(
                documents=documents, queries=None, metadata=metadata
            )
            catalog.save_dataset(dataset)

        # Load catalog
        catalog = Catalog(base_path=catalog_path)
        catalog.load()

        assert len(catalog.datasets) == num_datasets

        # List datasets
        dataset_names = catalog.list_datasets(as_df=False)
        assert len(dataset_names) == num_datasets

    def test_network_retry_scenario(self, tmpdir):
        """Test simulated network retry scenario"""
        dataset_path = str(tmpdir.mkdir("dataset"))
        documents_path = os.path.join(dataset_path, "documents")
        os.makedirs(documents_path)

        documents_data = pd.DataFrame(
            [{"id": "1", "values": [0.1, 0.2], "metadata": {"key": "value"}}]
        )
        documents_data.to_parquet(os.path.join(documents_path, "part-0.parquet"))

        metadata = {
            "name": "test",
            "created_at": "2021-01-01 00:00:00.000000",
            "documents": 1,
            "queries": 0,
            "dense_model": {"name": "ada2", "dimension": 2},
        }
        with open(os.path.join(dataset_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        # Simulate intermittent network failures
        call_count = [0]

        def flaky_open(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise IOError("Network error")
            # Use real implementation
            from fsspec.implementations.local import LocalFileSystem

            return LocalFileSystem().open(*args, **kwargs)

        from fsspec.implementations.local import LocalFileSystem

        fs = LocalFileSystem()

        # Mock open to be flaky
        with patch.object(fs, "open", side_effect=flaky_open):
            # First attempt should fail
            with pytest.raises(IOError):
                DatasetFSReader.read_metadata(fs, dataset_path)

            # Second attempt should also fail
            with pytest.raises(IOError):
                DatasetFSReader.read_metadata(fs, dataset_path)

    def test_dataset_schema_evolution(self, tmpdir):
        """Test handling datasets with schema changes"""
        dataset_path = str(tmpdir.mkdir("dataset"))
        documents_path = os.path.join(dataset_path, "documents")
        os.makedirs(documents_path)

        # Create parquet with extra column
        documents_data = pd.DataFrame(
            [
                {
                    "id": "1",
                    "values": [0.1, 0.2],
                    "metadata": {"key": "value"},
                    "extra_column": "extra_data",  # Not in schema
                }
            ]
        )
        documents_data.to_parquet(os.path.join(documents_path, "part-0.parquet"))

        metadata = {
            "name": "test",
            "created_at": "2021-01-01 00:00:00.000000",
            "documents": 1,
            "queries": 0,
            "dense_model": {"name": "ada2", "dimension": 2},
        }
        with open(os.path.join(dataset_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        # Should load successfully, extra column is allowed
        ds = Dataset.from_path(dataset_path)
        assert len(ds.documents) == 1

    def test_inconsistent_parquet_schemas(self, tmpdir):
        """Test loading dataset with inconsistent schemas across parquet files"""
        dataset_path = str(tmpdir.mkdir("dataset"))
        documents_path = os.path.join(dataset_path, "documents")
        os.makedirs(documents_path)

        # First parquet with standard schema
        data1 = pd.DataFrame(
            [{"id": "1", "values": [0.1, 0.2], "metadata": {"key": "value"}}]
        )
        data1.to_parquet(os.path.join(documents_path, "part-0.parquet"))

        # Second parquet with extra column
        data2 = pd.DataFrame(
            [
                {
                    "id": "2",
                    "values": [0.3, 0.4],
                    "metadata": {"key": "value2"},
                    "extra_field": "extra",
                }
            ]
        )
        data2.to_parquet(os.path.join(documents_path, "part-1.parquet"))

        metadata = {
            "name": "test",
            "created_at": "2021-01-01 00:00:00.000000",
            "documents": 2,
            "queries": 0,
            "dense_model": {"name": "ada2", "dimension": 2},
        }
        with open(os.path.join(dataset_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        # Should handle schema differences gracefully
        ds = Dataset.from_path(dataset_path)
        assert len(ds.documents) == 2

    def test_resource_cleanup_on_error(self, tmpdir):
        """Test that resources are properly cleaned up on error"""
        dataset_path = str(tmpdir.mkdir("dataset"))

        documents = pd.DataFrame(
            [{"id": "1", "values": [0.1, 0.2], "metadata": {"key": "value"}}]
        )

        metadata = DatasetMetadata(
            name="test",
            created_at="2021-01-01 00:00:00.000000",
            documents=1,
            queries=0,
            dense_model=DenseModelMetadata(name="ada2", dimension=2),
        )

        dataset = Dataset.from_pandas(
            documents=documents, queries=None, metadata=metadata
        )

        # Mock to cause error during write
        from fsspec.implementations.local import LocalFileSystem

        with patch(
            "pinecone_datasets.dataset_fswriter.get_cloud_fs",
            return_value=LocalFileSystem(),
        ):
            with patch("builtins.open", side_effect=IOError("Simulated error")):
                with pytest.raises(IOError):
                    DatasetFSWriter.write_dataset(dataset_path, dataset)

        # Original dataset should still be intact
        assert dataset.documents is not None
        assert len(dataset.documents) == 1

    def test_unicode_in_metadata(self, tmpdir):
        """Test handling unicode characters in metadata"""
        dataset_path = str(tmpdir.mkdir("dataset"))
        documents_path = os.path.join(dataset_path, "documents")
        os.makedirs(documents_path)

        # Documents with unicode in metadata
        documents_data = pd.DataFrame(
            [
                {
                    "id": "1",
                    "values": [0.1, 0.2],
                    "metadata": {"text": "日本語", "emoji": "🎉", "chinese": "中文"},
                }
            ]
        )
        documents_data.to_parquet(os.path.join(documents_path, "part-0.parquet"))

        metadata = {
            "name": "unicode_dataset",
            "created_at": "2021-01-01 00:00:00.000000",
            "documents": 1,
            "queries": 0,
            "dense_model": {"name": "ada2", "dimension": 2},
            "description": "データセット with unicode 🌟",
        }
        with open(os.path.join(dataset_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, ensure_ascii=False)

        # Should load successfully
        ds = Dataset.from_path(dataset_path)
        assert len(ds.documents) == 1
        assert ds.metadata.description == "データセット with unicode 🌟"
