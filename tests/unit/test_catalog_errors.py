import pytest
import json
import os
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from pinecone_datasets.catalog import Catalog
from pinecone_datasets import Dataset, DatasetMetadata, DenseModelMetadata


class TestCatalogErrorPaths:
    """Test error handling in Catalog"""

    def test_load_catalog_network_error(self):
        """Test loading catalog with network error"""
        catalog = Catalog(base_path="gs://test-bucket/catalog")

        mock_fs = Mock()
        mock_fs.glob.side_effect = IOError("Network connection failed")

        with patch("pinecone_datasets.catalog.get_cloud_fs", return_value=mock_fs):
            with pytest.raises(IOError):
                catalog.load()

    def test_load_catalog_invalid_json(self, tmpdir):
        """Test loading catalog with corrupted metadata.json"""
        catalog_path = str(tmpdir.mkdir("catalog"))
        dataset_path = os.path.join(catalog_path, "dataset1")

        os.makedirs(dataset_path)

        # Write invalid JSON
        with open(os.path.join(dataset_path, "metadata.json"), "w") as f:
            f.write("{invalid json content")

        catalog = Catalog(base_path=catalog_path)

        # Should warn and skip the invalid dataset
        with pytest.warns(UserWarning, match="Not a JSON"):
            result = catalog.load()
            assert len(result.datasets) == 0

    def test_load_catalog_missing_required_fields(self, tmpdir):
        """Test loading catalog with metadata missing required fields"""
        catalog_path = str(tmpdir.mkdir("catalog"))
        dataset_path = os.path.join(catalog_path, "dataset1")

        os.makedirs(dataset_path)

        # Write metadata with missing required field
        incomplete_metadata = {
            "created_at": "2021-01-01 00:00:00.000000",
            "documents": 10,
            "queries": 5,
        }

        with open(os.path.join(dataset_path, "metadata.json"), "w") as f:
            json.dump(incomplete_metadata, f)

        catalog = Catalog(base_path=catalog_path)

        # Should warn and skip the invalid dataset
        with pytest.warns(UserWarning, match="is not valid"):
            result = catalog.load()
            assert len(result.datasets) == 0

    def test_load_catalog_permission_denied(self):
        """Test loading catalog with permission denied"""
        catalog = Catalog(base_path="/restricted/path")

        mock_fs = Mock()
        mock_fs.glob.side_effect = PermissionError("Permission denied")

        with patch("pinecone_datasets.catalog.get_cloud_fs", return_value=mock_fs):
            with pytest.raises(PermissionError):
                catalog.load()

    def test_load_catalog_empty_catalog(self, tmpdir):
        """Test loading empty catalog"""
        catalog_path = str(tmpdir.mkdir("catalog"))
        catalog = Catalog(base_path=catalog_path)

        result = catalog.load()
        assert len(result.datasets) == 0

    def test_load_catalog_no_metadata_files(self, tmpdir):
        """Test loading catalog with directories but no metadata.json"""
        catalog_path = str(tmpdir.mkdir("catalog"))
        # Create directory without metadata.json
        dataset_path = os.path.join(catalog_path, "dataset1")

        os.makedirs(dataset_path)

        catalog = Catalog(base_path=catalog_path)
        result = catalog.load()
        assert len(result.datasets) == 0

    def test_list_datasets_auto_load_on_empty(self, tmpdir):
        """Test that list_datasets auto-loads when datasets is empty"""
        catalog_path = str(tmpdir.mkdir("catalog"))
        dataset_path = os.path.join(catalog_path, "dataset1")

        os.makedirs(dataset_path)

        # Create valid metadata
        metadata = {
            "name": "dataset1",
            "created_at": "2021-01-01 00:00:00.000000",
            "documents": 10,
            "queries": 5,
            "dense_model": {"name": "ada2", "dimension": 2},
        }

        with open(os.path.join(dataset_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        catalog = Catalog(base_path=catalog_path)
        # Don't call load() first
        result = catalog.list_datasets(as_df=False)
        assert "dataset1" in result

    def test_load_dataset_nonexistent(self, tmpdir):
        """Test loading nonexistent dataset"""
        catalog_path = str(tmpdir.mkdir("catalog"))
        catalog = Catalog(base_path=catalog_path)

        with pytest.raises(FileNotFoundError):
            catalog.load_dataset("nonexistent_dataset")

    def test_load_dataset_corrupted_data(self, tmpdir):
        """Test loading dataset with corrupted data"""
        catalog_path = str(tmpdir.mkdir("catalog"))
        dataset_path = os.path.join(catalog_path, "dataset1")

        os.makedirs(dataset_path)

        # Create metadata
        metadata = {
            "name": "dataset1",
            "created_at": "2021-01-01 00:00:00.000000",
            "documents": 10,
            "queries": 5,
            "dense_model": {"name": "ada2", "dimension": 2},
        }

        with open(os.path.join(dataset_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        # Create corrupted parquet file
        documents_path = os.path.join(dataset_path, "documents")
        os.makedirs(documents_path)
        with open(os.path.join(documents_path, "part-0.parquet"), "w") as f:
            f.write("corrupted data")

        catalog = Catalog(base_path=catalog_path)

        # Should raise error when trying to load corrupted dataset
        with pytest.raises(Exception):
            ds = catalog.load_dataset("dataset1")
            # Access documents to trigger loading
            _ = ds.documents

    def test_save_dataset_invalid_path(self):
        """Test saving dataset to invalid path"""
        catalog = Catalog(base_path="/invalid/\x00/path")

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

        with pytest.raises(Exception):
            catalog.save_dataset(dataset)

    def test_save_dataset_permission_denied(self, tmpdir):
        """Test saving dataset with permission denied"""
        catalog_path = str(tmpdir.mkdir("catalog"))
        catalog = Catalog(base_path=catalog_path)

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

        # Mock get_cloud_fs to return filesystem that raises permission error
        mock_fs = Mock()
        mock_fs.makedirs.side_effect = PermissionError("Permission denied")

        with patch(
            "pinecone_datasets.dataset_fswriter.get_cloud_fs", return_value=mock_fs
        ):
            with pytest.raises(PermissionError):
                catalog.save_dataset(dataset)

    def test_save_dataset_disk_full(self, tmpdir):
        """Test saving dataset when disk is full"""
        catalog_path = str(tmpdir.mkdir("catalog"))
        catalog = Catalog(base_path=catalog_path)

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
                    catalog.save_dataset(dataset)

    def test_load_catalog_mixed_valid_invalid_datasets(self, tmpdir):
        """Test loading catalog with mix of valid and invalid datasets"""
        catalog_path = str(tmpdir.mkdir("catalog"))

        # Create valid dataset
        dataset1_path = dataset_path = os.path.join(catalog_path, "dataset1")
        os.makedirs(dataset_path)
        valid_metadata = {
            "name": "dataset1",
            "created_at": "2021-01-01 00:00:00.000000",
            "documents": 10,
            "queries": 5,
            "dense_model": {"name": "ada2", "dimension": 2},
        }
        with open(os.path.join(dataset1_path, "metadata.json"), "w") as f:
            json.dump(valid_metadata, f)

        # Create invalid dataset (missing required field)
        dataset2_path = dataset_path = os.path.join(catalog_path, "dataset2")
        os.makedirs(dataset_path)
        invalid_metadata = {
            "created_at": "2021-01-01 00:00:00.000000",
            "documents": 10,
            "queries": 5,
        }
        with open(os.path.join(dataset2_path, "metadata.json"), "w") as f:
            json.dump(invalid_metadata, f)

        catalog = Catalog(base_path=catalog_path)

        # Should load only valid datasets and warn about invalid ones
        with pytest.warns(UserWarning):
            result = catalog.load()
            assert len(result.datasets) == 1
            assert result.datasets[0].name == "dataset1"

    def test_load_catalog_file_open_error(self):
        """Test loading catalog when file cannot be opened"""
        catalog = Catalog(base_path="gs://test-bucket/catalog")

        mock_fs = Mock()
        mock_fs.glob.return_value = ["gs://test-bucket/catalog/dataset1/metadata.json"]
        mock_fs.open.side_effect = IOError("Cannot open file")

        with patch("pinecone_datasets.catalog.get_cloud_fs", return_value=mock_fs):
            with pytest.raises(IOError):
                catalog.load()

    def test_list_datasets_as_dataframe(self, tmpdir):
        """Test listing datasets as DataFrame"""
        catalog_path = str(tmpdir.mkdir("catalog"))
        dataset_path = os.path.join(catalog_path, "dataset1")

        os.makedirs(dataset_path)

        metadata = {
            "name": "dataset1",
            "created_at": "2021-01-01 00:00:00.000000",
            "documents": 10,
            "queries": 5,
            "dense_model": {"name": "ada2", "dimension": 2},
        }

        with open(os.path.join(dataset_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        catalog = Catalog(base_path=catalog_path)
        result = catalog.list_datasets(as_df=True)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert "name" in result.columns

    def test_catalog_default_base_path(self):
        """Test catalog uses default base path from environment or config"""
        catalog = Catalog()
        assert catalog.base_path is not None

    def test_catalog_custom_base_path(self):
        """Test catalog with custom base path"""
        custom_path = "/custom/path"
        catalog = Catalog(base_path=custom_path)
        assert catalog.base_path == custom_path

    def test_load_dataset_missing_metadata(self, tmpdir):
        """Test loading dataset with missing metadata.json"""
        catalog_path = str(tmpdir.mkdir("catalog"))
        dataset_path = os.path.join(catalog_path, "dataset1")

        os.makedirs(dataset_path)

        # Create only documents directory, no metadata
        os.makedirs(os.path.join(dataset_path, "documents"))

        catalog = Catalog(base_path=catalog_path)

        with pytest.raises(Exception):
            ds = catalog.load_dataset("dataset1")
            # Access metadata to trigger loading
            _ = ds.metadata

    def test_load_dataset_network_timeout(self):
        """Test loading dataset with network timeout"""
        catalog = Catalog(base_path="gs://test-bucket/catalog")

        mock_ds = Mock(spec=Dataset)
        mock_ds.from_path.side_effect = TimeoutError("Network timeout")

        with patch(
            "pinecone_datasets.catalog.Dataset.from_path",
            side_effect=TimeoutError("Network timeout"),
        ):
            with pytest.raises(TimeoutError):
                catalog.load_dataset("dataset1")

    def test_save_dataset_network_failure(self, tmpdir):
        """Test saving dataset with network failure"""
        catalog_path = "gs://test-bucket/catalog"
        catalog = Catalog(base_path=catalog_path)

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

        # Mock DatasetFSWriter to raise network error
        with patch(
            "pinecone_datasets.catalog.DatasetFSWriter.write_dataset",
            side_effect=IOError("Network error"),
        ):
            with pytest.raises(IOError):
                catalog.save_dataset(dataset)

    def test_load_catalog_empty_metadata_file(self, tmpdir):
        """Test loading catalog with empty metadata file"""
        catalog_path = str(tmpdir.mkdir("catalog"))
        dataset_path = os.path.join(catalog_path, "dataset1")

        os.makedirs(dataset_path)

        # Create empty metadata file
        with open(os.path.join(dataset_path, "metadata.json"), "w") as f:
            f.write("")

        catalog = Catalog(base_path=catalog_path)

        # Should warn and skip the invalid dataset
        with pytest.warns(UserWarning, match="Not a JSON"):
            result = catalog.load()
            assert len(result.datasets) == 0

    def test_load_catalog_metadata_wrong_type(self, tmpdir):
        """Test loading catalog with metadata of wrong type (list instead of dict)"""
        catalog_path = str(tmpdir.mkdir("catalog"))
        dataset_path = os.path.join(catalog_path, "dataset1")

        os.makedirs(dataset_path)

        # Write list instead of dict
        with open(os.path.join(dataset_path, "metadata.json"), "w") as f:
            json.dump(["not", "a", "dict"], f)

        catalog = Catalog(base_path=catalog_path)

        # TypeError is raised when trying to unpack non-dict as **kwargs
        with pytest.raises(TypeError):
            catalog.load()

    def test_save_dataset_writes_to_correct_path(self, tmpdir):
        """Test that save_dataset writes to correct path with dataset name"""
        catalog_path = str(tmpdir.mkdir("catalog"))
        catalog = Catalog(base_path=catalog_path)

        documents = pd.DataFrame(
            [{"id": "1", "values": [0.1, 0.2], "metadata": {"key": "value"}}]
        )

        metadata = DatasetMetadata(
            name="my_dataset",
            created_at="2021-01-01 00:00:00.000000",
            documents=1,
            queries=0,
            dense_model=DenseModelMetadata(name="ada2", dimension=2),
        )

        dataset = Dataset.from_pandas(
            documents=documents, queries=None, metadata=metadata
        )
        catalog.save_dataset(dataset)

        # Verify dataset was saved to correct path
        expected_path = os.path.join(catalog_path, "my_dataset")
        assert os.path.exists(expected_path)
        assert os.path.exists(os.path.join(expected_path, "metadata.json"))
        assert os.path.exists(os.path.join(expected_path, "documents"))

    def test_load_catalog_with_env_variable(self, tmpdir, monkeypatch):
        """Test catalog uses environment variable for base path"""
        catalog_path = str(tmpdir.mkdir("catalog"))
        monkeypatch.setenv("DATASETS_CATALOG_BASEPATH", catalog_path)

        catalog = Catalog()
        assert catalog.base_path == catalog_path
