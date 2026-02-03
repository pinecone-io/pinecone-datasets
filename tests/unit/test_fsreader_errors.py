import json
import os
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from pinecone_datasets.dataset_fsreader import DatasetFSReader


class TestFSReaderErrorPaths:
    """Test error handling in DatasetFSReader"""

    def test_read_metadata_file_not_found(self, tmpdir):
        """Test reading metadata from non-existent file"""
        dataset_path = str(tmpdir.mkdir("dataset"))

        mock_fs = Mock()
        mock_fs.open.side_effect = FileNotFoundError("metadata.json not found")

        with pytest.raises(FileNotFoundError):
            DatasetFSReader.read_metadata(mock_fs, dataset_path)

    def test_read_metadata_invalid_json(self, tmpdir):
        """Test reading metadata with corrupted JSON"""
        dataset_path = str(tmpdir.mkdir("dataset"))
        metadata_path = os.path.join(dataset_path, "metadata.json")

        with open(metadata_path, "w") as f:
            f.write("{invalid json content")

        from fsspec.implementations.local import LocalFileSystem

        fs = LocalFileSystem()

        with pytest.raises(json.JSONDecodeError):
            DatasetFSReader.read_metadata(fs, dataset_path)

    def test_read_metadata_missing_required_fields(self, tmpdir):
        """Test reading metadata with missing required fields"""
        dataset_path = str(tmpdir.mkdir("dataset"))
        metadata_path = os.path.join(dataset_path, "metadata.json")

        # Missing required field 'name'
        incomplete_metadata = {
            "created_at": "2021-01-01 00:00:00.000000",
            "documents": 10,
            "queries": 5,
        }

        with open(metadata_path, "w") as f:
            json.dump(incomplete_metadata, f)

        from fsspec.implementations.local import LocalFileSystem

        fs = LocalFileSystem()

        with pytest.raises(Exception):  # Pydantic ValidationError
            DatasetFSReader.read_metadata(fs, dataset_path)

    def test_read_metadata_invalid_field_types(self, tmpdir):
        """Test reading metadata with invalid field types"""
        dataset_path = str(tmpdir.mkdir("dataset"))
        metadata_path = os.path.join(dataset_path, "metadata.json")

        # documents should be int, not string
        invalid_metadata = {
            "name": "test",
            "created_at": "2021-01-01 00:00:00.000000",
            "documents": "not_a_number",
            "queries": 5,
            "dense_model": {"name": "ada2", "dimension": 2},
        }

        with open(metadata_path, "w") as f:
            json.dump(invalid_metadata, f)

        from fsspec.implementations.local import LocalFileSystem

        fs = LocalFileSystem()

        with pytest.raises(Exception):  # Pydantic ValidationError
            DatasetFSReader.read_metadata(fs, dataset_path)

    def test_read_documents_no_parquet_files(self, tmpdir):
        """Test reading documents when no parquet files exist"""
        dataset_path = str(tmpdir.mkdir("dataset"))
        documents_path = os.path.join(dataset_path, "documents")
        os.makedirs(documents_path)

        from fsspec.implementations.local import LocalFileSystem

        fs = LocalFileSystem()

        with pytest.raises(ValueError, match="No parquet files found"):
            DatasetFSReader.read_documents(fs, dataset_path)

    def test_read_documents_missing_required_column(self, tmpdir):
        """Test reading documents with missing required column"""
        dataset_path = str(tmpdir.mkdir("dataset"))
        documents_path = os.path.join(dataset_path, "documents")
        os.makedirs(documents_path)

        # Missing required 'id' column
        documents_data = pd.DataFrame(
            [{"values": [0.1, 0.2, 0.3], "metadata": {"title": "test"}}]
        )

        documents_data.to_parquet(os.path.join(documents_path, "part-0.parquet"))

        from fsspec.implementations.local import LocalFileSystem

        fs = LocalFileSystem()

        with pytest.raises(ValueError, match="not matching Pinecone Datasets Schema"):
            DatasetFSReader.read_documents(fs, dataset_path)

    def test_read_queries_missing_required_column(self, tmpdir):
        """Test reading queries with missing required column"""
        dataset_path = str(tmpdir.mkdir("dataset"))
        queries_path = os.path.join(dataset_path, "queries")
        os.makedirs(queries_path)

        # Missing required 'vector' column
        queries_data = pd.DataFrame([{"filter": {"key": "value"}, "top_k": 10}])

        queries_data.to_parquet(os.path.join(queries_path, "part-0.parquet"))

        from fsspec.implementations.local import LocalFileSystem

        fs = LocalFileSystem()

        with pytest.raises(ValueError, match="not matching Pinecone Datasets Schema"):
            DatasetFSReader.read_queries(fs, dataset_path)

    def test_convert_metadata_invalid_type(self):
        """Test metadata conversion with invalid type"""
        with pytest.raises(TypeError, match="metadata must be a string or dict"):
            DatasetFSReader._convert_metadata_from_json_to_dict(123)

    def test_convert_metadata_invalid_json_string(self):
        """Test metadata conversion with invalid JSON string"""
        with pytest.raises(json.JSONDecodeError):
            DatasetFSReader._convert_metadata_from_json_to_dict("{invalid json")

    def test_read_documents_corrupted_parquet(self, tmpdir):
        """Test reading documents with corrupted parquet file"""
        dataset_path = str(tmpdir.mkdir("dataset"))
        documents_path = os.path.join(dataset_path, "documents")
        os.makedirs(documents_path)

        # Write corrupted parquet file
        corrupted_file = os.path.join(documents_path, "part-0.parquet")
        with open(corrupted_file, "w") as f:
            f.write("This is not a valid parquet file")

        from fsspec.implementations.local import LocalFileSystem

        fs = LocalFileSystem()

        with pytest.raises(Exception):  # pyarrow will raise an error
            DatasetFSReader.read_documents(fs, dataset_path)

    def test_read_queries_corrupted_parquet(self, tmpdir):
        """Test reading queries with corrupted parquet file"""
        dataset_path = str(tmpdir.mkdir("dataset"))
        queries_path = os.path.join(dataset_path, "queries")
        os.makedirs(queries_path)

        # Write corrupted parquet file
        corrupted_file = os.path.join(queries_path, "part-0.parquet")
        with open(corrupted_file, "w") as f:
            f.write("This is not a valid parquet file")

        from fsspec.implementations.local import LocalFileSystem

        fs = LocalFileSystem()

        with pytest.raises(Exception):  # pyarrow will raise an error
            DatasetFSReader.read_queries(fs, dataset_path)

    def test_read_documents_empty_parquet(self, tmpdir):
        """Test reading documents from empty parquet files"""
        dataset_path = str(tmpdir.mkdir("dataset"))
        documents_path = os.path.join(dataset_path, "documents")
        os.makedirs(documents_path)

        # Create valid but empty dataframe
        empty_df = pd.DataFrame(columns=["id", "values", "metadata"])
        empty_df.to_parquet(os.path.join(documents_path, "part-0.parquet"))

        from fsspec.implementations.local import LocalFileSystem

        fs = LocalFileSystem()

        # Should not raise error but return valid dataframe with proper columns
        result = DatasetFSReader.read_documents(fs, dataset_path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert "id" in result.columns
        assert "values" in result.columns

    def test_read_documents_path_does_not_exist(self, tmpdir):
        """Test reading documents when path doesn't exist"""
        dataset_path = str(tmpdir.mkdir("dataset"))

        from fsspec.implementations.local import LocalFileSystem

        fs = LocalFileSystem()

        # Should return empty dataframe with warning
        with pytest.warns(UserWarning, match="No data found"):
            result = DatasetFSReader.read_documents(fs, dataset_path)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    def test_read_queries_path_does_not_exist(self, tmpdir):
        """Test reading queries when path doesn't exist"""
        dataset_path = str(tmpdir.mkdir("dataset"))

        from fsspec.implementations.local import LocalFileSystem

        fs = LocalFileSystem()

        # Should return empty dataframe with warning
        with pytest.warns(UserWarning, match="No data found"):
            result = DatasetFSReader.read_queries(fs, dataset_path)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    def test_read_documents_permission_denied(self, tmpdir):
        """Test reading documents with permission denied"""
        mock_fs = Mock()
        mock_fs.glob.return_value = ["path/to/file.parquet"]
        mock_fs.exists.return_value = True

        # Simulate permission error
        with patch("pyarrow.parquet.read_pandas") as mock_read:
            mock_read.side_effect = PermissionError("Permission denied")

            with pytest.raises(PermissionError):
                DatasetFSReader.read_documents(mock_fs, "/test/path")

    def test_read_metadata_permission_denied(self, tmpdir):
        """Test reading metadata with permission denied"""
        mock_fs = Mock()
        mock_fs.open.side_effect = PermissionError("Permission denied")

        with pytest.raises(PermissionError):
            DatasetFSReader.read_metadata(mock_fs, "/test/path")

    def test_read_documents_network_error(self):
        """Test reading documents with simulated network error"""
        mock_fs = Mock()
        mock_fs.glob.return_value = ["gs://bucket/dataset/documents/part-0.parquet"]
        mock_fs.exists.return_value = True

        # Simulate network error by patching at the read level
        with patch("pyarrow.parquet.read_pandas") as mock_read:
            mock_read.side_effect = OSError("Network connection failed")
            # Patch get_cached_path to bypass caching for this test
            with patch(
                "pinecone_datasets.dataset_fsreader.get_cached_path",
                side_effect=lambda p, fs: p,
            ):
                with pytest.raises(OSError):
                    DatasetFSReader.read_documents(mock_fs, "gs://bucket/dataset")

    def test_read_metadata_network_error(self):
        """Test reading metadata with simulated network error"""
        mock_fs = Mock()
        mock_fs.open.side_effect = OSError("Network connection failed")

        with pytest.raises(OSError):
            DatasetFSReader.read_metadata(mock_fs, "gs://bucket/dataset")

    def test_convert_metadata_null_value(self):
        """Test metadata conversion with None value"""
        result = DatasetFSReader._convert_metadata_from_json_to_dict(None)
        assert result is None

    def test_convert_metadata_valid_dict(self):
        """Test metadata conversion with valid dict"""
        test_dict = {"key": "value", "number": 123}
        result = DatasetFSReader._convert_metadata_from_json_to_dict(test_dict)
        assert result == test_dict

    def test_convert_metadata_valid_json_string(self):
        """Test metadata conversion with valid JSON string"""
        test_dict = {"key": "value", "number": 123}
        json_str = json.dumps(test_dict)
        result = DatasetFSReader._convert_metadata_from_json_to_dict(json_str)
        assert result == test_dict

    def test_read_documents_multiple_parquet_files_one_corrupted(self, tmpdir):
        """Test reading when one of multiple parquet files is corrupted"""
        dataset_path = str(tmpdir.mkdir("dataset"))
        documents_path = os.path.join(dataset_path, "documents")
        os.makedirs(documents_path)

        # Write one valid parquet file
        valid_data = pd.DataFrame(
            [{"id": "1", "values": [0.1, 0.2], "metadata": {"title": "test"}}]
        )
        valid_data.to_parquet(os.path.join(documents_path, "part-0.parquet"))

        # Write one corrupted parquet file
        with open(os.path.join(documents_path, "part-1.parquet"), "w") as f:
            f.write("corrupted data")

        from fsspec.implementations.local import LocalFileSystem

        fs = LocalFileSystem()

        # Should raise error when trying to read corrupted file
        with pytest.raises(Exception):
            DatasetFSReader.read_documents(fs, dataset_path)

    def test_read_metadata_empty_file(self, tmpdir):
        """Test reading metadata from empty file"""
        dataset_path = str(tmpdir.mkdir("dataset"))
        metadata_path = os.path.join(dataset_path, "metadata.json")

        # Create empty file
        with open(metadata_path, "w") as f:
            f.write("")

        from fsspec.implementations.local import LocalFileSystem

        fs = LocalFileSystem()

        with pytest.raises(json.JSONDecodeError):
            DatasetFSReader.read_metadata(fs, dataset_path)

    def test_read_documents_with_invalid_metadata_json_in_parquet(self, tmpdir):
        """Test reading documents when metadata field contains invalid JSON"""
        dataset_path = str(tmpdir.mkdir("dataset"))
        documents_path = os.path.join(dataset_path, "documents")
        os.makedirs(documents_path)

        # Create parquet with metadata as invalid JSON string
        documents_data = pd.DataFrame(
            [
                {
                    "id": "1",
                    "values": [0.1, 0.2],
                    "metadata": "{invalid json",  # Invalid JSON string
                }
            ]
        )
        documents_data.to_parquet(os.path.join(documents_path, "part-0.parquet"))

        from fsspec.implementations.local import LocalFileSystem

        fs = LocalFileSystem()

        # Should raise JSONDecodeError when trying to convert metadata
        with pytest.raises(json.JSONDecodeError):
            DatasetFSReader.read_documents(fs, dataset_path)
