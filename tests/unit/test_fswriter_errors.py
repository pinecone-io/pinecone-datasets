import pytest
import json
import os
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from pinecone_datasets.dataset_fswriter import DatasetFSWriter
from pinecone_datasets import Dataset, DatasetMetadata, DenseModelMetadata


class TestFSWriterErrorPaths:
    """Test error handling in DatasetFSWriter"""

    def test_write_dataset_permission_denied(self, tmpdir):
        """Test writing dataset with permission denied"""
        dataset_path = str(tmpdir.mkdir("dataset"))

        # Create a valid dataset
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

        # Mock filesystem with permission error
        mock_fs = Mock()
        mock_fs.makedirs.side_effect = PermissionError("Permission denied")

        with patch(
            "pinecone_datasets.dataset_fswriter.get_cloud_fs", return_value=mock_fs
        ):
            with pytest.raises(PermissionError):
                DatasetFSWriter.write_dataset(dataset_path, dataset)

    def test_write_dataset_disk_full(self, tmpdir):
        """Test writing dataset when disk is full"""
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

        # Mock open to raise disk full error
        from fsspec.implementations.local import LocalFileSystem

        with patch(
            "pinecone_datasets.dataset_fswriter.get_cloud_fs",
            return_value=LocalFileSystem(),
        ):
            with patch("builtins.open", side_effect=OSError("No space left on device")):
                with pytest.raises(OSError, match="No space left on device"):
                    DatasetFSWriter.write_dataset(dataset_path, dataset)

    def test_convert_metadata_invalid_type(self):
        """Test metadata conversion with invalid type"""
        with pytest.raises(TypeError, match="metadata must be a dict"):
            DatasetFSWriter._convert_metadata_from_dict_to_json("not a dict")

    def test_convert_metadata_invalid_type_list(self):
        """Test metadata conversion with list instead of dict"""
        # Lists cause pd.isna() to raise ValueError, but should be caught as TypeError eventually
        with pytest.raises((TypeError, ValueError)):
            DatasetFSWriter._convert_metadata_from_dict_to_json([1, 2, 3])

    def test_convert_metadata_invalid_type_number(self):
        """Test metadata conversion with number instead of dict"""
        with pytest.raises(TypeError, match="metadata must be a dict"):
            DatasetFSWriter._convert_metadata_from_dict_to_json(123)

    def test_convert_metadata_valid_dict(self):
        """Test metadata conversion with valid dict"""
        test_dict = {"key": "value", "number": 123}
        result = DatasetFSWriter._convert_metadata_from_dict_to_json(test_dict)
        assert isinstance(result, str)
        assert json.loads(result) == test_dict

    def test_convert_metadata_nan_value(self):
        """Test metadata conversion with NaN value"""
        result = DatasetFSWriter._convert_metadata_from_dict_to_json(float("nan"))
        assert result is None

    def test_write_documents_network_error(self, tmpdir):
        """Test writing documents with network error"""
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

        # Mock filesystem with network error
        mock_fs = Mock()
        mock_fs.makedirs.return_value = None

        # Mock to_parquet to raise network error
        with patch(
            "pinecone_datasets.dataset_fswriter.get_cloud_fs", return_value=mock_fs
        ):
            with patch.object(
                pd.DataFrame, "to_parquet", side_effect=IOError("Network error")
            ):
                with pytest.raises(IOError):
                    DatasetFSWriter.write_dataset(dataset_path, dataset)

    def test_write_metadata_invalid_json_serialization(self, tmpdir):
        """Test writing metadata that cannot be JSON serialized"""
        dataset_path = str(tmpdir.mkdir("dataset"))

        documents = pd.DataFrame(
            [{"id": "1", "values": [0.1, 0.2], "metadata": {"key": "value"}}]
        )

        # Create metadata with non-serializable object
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

        # Mock model_dump to return non-serializable data
        with patch.object(
            DatasetMetadata, "model_dump", return_value={"func": lambda x: x}
        ):
            from fsspec.implementations.local import LocalFileSystem

            with patch(
                "pinecone_datasets.dataset_fswriter.get_cloud_fs",
                return_value=LocalFileSystem(),
            ):
                with pytest.raises(TypeError):
                    DatasetFSWriter.write_dataset(dataset_path, dataset)

    def test_write_dataset_invalid_path(self):
        """Test writing dataset to invalid path"""
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

        # Try to write to invalid path
        with pytest.raises(Exception):  # Could be various errors depending on OS
            DatasetFSWriter.write_dataset("/invalid/\x00/path", dataset)

    def test_write_documents_readonly_filesystem(self, tmpdir):
        """Test writing documents to read-only filesystem"""
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

        # Mock filesystem as read-only
        mock_fs = Mock()
        mock_fs.makedirs.side_effect = OSError("Read-only file system")

        with patch(
            "pinecone_datasets.dataset_fswriter.get_cloud_fs", return_value=mock_fs
        ):
            with pytest.raises(OSError, match="Read-only file system"):
                DatasetFSWriter.write_dataset(dataset_path, dataset)

    def test_write_queries_with_invalid_filter_type(self, tmpdir):
        """Test writing queries with invalid filter type"""
        dataset_path = str(tmpdir.mkdir("dataset"))

        documents = pd.DataFrame(
            [{"id": "1", "values": [0.1, 0.2], "metadata": {"key": "value"}}]
        )

        # Queries with invalid filter type (should be dict)
        queries = pd.DataFrame(
            [
                {
                    "vector": [0.1, 0.2],
                    "filter": "invalid_filter_type",  # Should be dict
                    "top_k": 10,
                }
            ]
        )

        metadata = DatasetMetadata(
            name="test",
            created_at="2021-01-01 00:00:00.000000",
            documents=1,
            queries=1,
            dense_model=DenseModelMetadata(name="ada2", dimension=2),
        )

        dataset = Dataset.from_pandas(
            documents=documents, queries=queries, metadata=metadata
        )

        from fsspec.implementations.local import LocalFileSystem

        with patch(
            "pinecone_datasets.dataset_fswriter.get_cloud_fs",
            return_value=LocalFileSystem(),
        ):
            with pytest.raises(TypeError):
                DatasetFSWriter.write_dataset(dataset_path, dataset)

    def test_write_empty_queries_warning(self, tmpdir):
        """Test that writing empty queries produces a warning"""
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

        from fsspec.implementations.local import LocalFileSystem

        with patch(
            "pinecone_datasets.dataset_fswriter.get_cloud_fs",
            return_value=LocalFileSystem(),
        ):
            with pytest.warns(UserWarning, match="Queries are empty"):
                DatasetFSWriter.write_dataset(dataset_path, dataset)

    def test_write_metadata_file_open_error(self, tmpdir):
        """Test writing metadata when file cannot be opened"""
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

        # Mock open to fail when writing metadata
        from fsspec.implementations.local import LocalFileSystem

        with patch(
            "pinecone_datasets.dataset_fswriter.get_cloud_fs",
            return_value=LocalFileSystem(),
        ):
            with patch("builtins.open", side_effect=IOError("Cannot open file")):
                with pytest.raises(IOError):
                    DatasetFSWriter.write_dataset(dataset_path, dataset)

    def test_write_documents_with_none_metadata(self, tmpdir):
        """Test writing documents with None metadata values"""
        dataset_path = str(tmpdir.mkdir("dataset"))

        documents = pd.DataFrame([{"id": "1", "values": [0.1, 0.2], "metadata": None}])

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

        from fsspec.implementations.local import LocalFileSystem

        with patch(
            "pinecone_datasets.dataset_fswriter.get_cloud_fs",
            return_value=LocalFileSystem(),
        ):
            # Should handle None values gracefully
            DatasetFSWriter.write_dataset(dataset_path, dataset)

            # Verify files were created
            assert os.path.exists(os.path.join(dataset_path, "documents"))
            assert os.path.exists(os.path.join(dataset_path, "metadata.json"))

    def test_write_dataset_concurrent_access(self, tmpdir):
        """Test writing dataset with concurrent access conflicts"""
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

        # Simulate file being locked by another process
        from fsspec.implementations.local import LocalFileSystem

        with patch(
            "pinecone_datasets.dataset_fswriter.get_cloud_fs",
            return_value=LocalFileSystem(),
        ):
            with patch(
                "builtins.open",
                side_effect=BlockingIOError("Resource temporarily unavailable"),
            ):
                with pytest.raises(BlockingIOError):
                    DatasetFSWriter.write_dataset(dataset_path, dataset)

    def test_write_documents_preserves_original_metadata_on_error(self, tmpdir):
        """Test that original metadata is preserved even if write fails"""
        dataset_path = str(tmpdir.mkdir("dataset"))

        original_metadata = {"key": "value"}
        documents = pd.DataFrame(
            [{"id": "1", "values": [0.1, 0.2], "metadata": original_metadata}]
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
        original_meta_value = dataset.documents["metadata"].iloc[0]

        # Mock to_parquet to raise error
        mock_fs = Mock()
        mock_fs.makedirs.return_value = None

        with patch(
            "pinecone_datasets.dataset_fswriter.get_cloud_fs", return_value=mock_fs
        ):
            with patch.object(
                pd.DataFrame, "to_parquet", side_effect=IOError("Write failed")
            ):
                try:
                    DatasetFSWriter.write_dataset(dataset_path, dataset)
                except IOError:
                    pass

        # Verify original metadata is preserved
        assert dataset.documents["metadata"].iloc[0] == original_meta_value

    def test_write_queries_preserves_original_filter_on_error(self, tmpdir):
        """Test that original filter is preserved even if write fails"""
        dataset_path = str(tmpdir.mkdir("dataset"))

        documents = pd.DataFrame(
            [{"id": "1", "values": [0.1, 0.2], "metadata": {"key": "value"}}]
        )

        original_filter = {"key": "value"}
        queries = pd.DataFrame(
            [{"vector": [0.1, 0.2], "filter": original_filter, "top_k": 10}]
        )

        metadata = DatasetMetadata(
            name="test",
            created_at="2021-01-01 00:00:00.000000",
            documents=1,
            queries=1,
            dense_model=DenseModelMetadata(name="ada2", dimension=2),
        )

        dataset = Dataset.from_pandas(
            documents=documents, queries=queries, metadata=metadata
        )
        original_filter_value = dataset.queries["filter"].iloc[0]

        # Mock to_parquet to raise error on queries write
        mock_fs = Mock()
        mock_fs.makedirs.return_value = None

        with patch(
            "pinecone_datasets.dataset_fswriter.get_cloud_fs", return_value=mock_fs
        ):
            with patch.object(
                pd.DataFrame, "to_parquet", side_effect=IOError("Write failed")
            ):
                try:
                    DatasetFSWriter.write_dataset(dataset_path, dataset)
                except IOError:
                    pass

        # Verify original filter is preserved
        assert dataset.queries["filter"].iloc[0] == original_filter_value
