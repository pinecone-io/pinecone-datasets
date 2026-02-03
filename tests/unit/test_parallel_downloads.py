"""
Unit tests for parallel download functionality.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, call

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pinecone_datasets import cfg
from pinecone_datasets.dataset_fsreader import DatasetFSReader


class TestParallelDownloads:
    """Test parallel download functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.original_max_workers = cfg.Cache.max_parallel_downloads

    def teardown_method(self):
        """Restore original configuration."""
        cfg.Cache.max_parallel_downloads = self.original_max_workers

    def test_parallel_download_configuration(self):
        """Test that max_parallel_downloads can be configured."""
        # Test default value
        assert cfg.Cache.max_parallel_downloads == 4

        # Test setting via attribute
        cfg.Cache.max_parallel_downloads = 8
        assert cfg.Cache.max_parallel_downloads == 8

    def test_parallel_download_environment_variable(self):
        """Test that max_parallel_downloads can be set via environment variable."""
        with patch.dict(os.environ, {"PINECONE_DATASETS_MAX_PARALLEL_DOWNLOADS": "10"}):
            # Need to reload the module for env var to take effect
            import importlib
            from pinecone_datasets import cfg as cfg_module

            importlib.reload(cfg_module)
            assert cfg_module.Cache.max_parallel_downloads == 10

    def test_download_and_read_parquet_extracts_file_index(self):
        """Test that _download_and_read_parquet extracts file index from filename."""
        # Create a temporary parquet file
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            df = pd.DataFrame(
                {
                    "id": ["1", "2"],
                    "values": [[0.1, 0.2], [0.3, 0.4]],
                    "sparse_values": [None, None],
                    "metadata": [None, None],
                    "blob": [None, None],
                }
            )

            # Write to parquet with numbered filename
            path = os.path.join(tmpdir, "0042.parquet")
            table = pa.Table.from_pandas(df)
            pq.write_table(table, path)

            # Mock filesystem
            from fsspec.implementations.local import LocalFileSystem

            fs = LocalFileSystem()

            # Test extraction
            file_index, result_df = DatasetFSReader._download_and_read_parquet(
                path=path, fs=fs, use_cache=False, protocol=None
            )

            assert file_index == 42
            assert len(result_df) == 2

    def test_download_and_read_parquet_handles_non_numeric_filenames(self):
        """Test that _download_and_read_parquet handles non-numeric filenames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            df = pd.DataFrame(
                {
                    "id": ["1", "2"],
                    "values": [[0.1, 0.2], [0.3, 0.4]],
                    "sparse_values": [None, None],
                    "metadata": [None, None],
                    "blob": [None, None],
                }
            )

            # Write to parquet with non-numeric filename
            path = os.path.join(tmpdir, "part-abc.parquet")
            table = pa.Table.from_pandas(df)
            pq.write_table(table, path)

            from fsspec.implementations.local import LocalFileSystem

            fs = LocalFileSystem()

            # Test extraction - should use hash fallback
            file_index, result_df = DatasetFSReader._download_and_read_parquet(
                path=path, fs=fs, use_cache=False, protocol=None
            )

            # File index should be consistent for same path
            assert isinstance(file_index, int)
            assert file_index == hash(path)

    def test_safe_read_from_path_sorts_by_file_index(self):
        """Test that files are read in correct order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dataset directory structure
            docs_dir = Path(tmpdir) / "test-dataset" / "documents"
            docs_dir.mkdir(parents=True)

            # Create multiple parquet files with different indices
            # Write them in reverse order to test sorting
            for i in [2, 0, 1]:
                df = pd.DataFrame(
                    {
                        "id": [f"id_{i}"],
                        "values": [[float(i)]],
                        "sparse_values": [None],
                        "metadata": [None],
                        "blob": [None],
                    }
                )
                path = docs_dir / f"{i:04d}.parquet"
                table = pa.Table.from_pandas(df)
                pq.write_table(table, str(path))

            from fsspec.implementations.local import LocalFileSystem

            fs = LocalFileSystem()

            # Read with parallel processing disabled
            cfg.Cache.max_parallel_downloads = 1
            result_df = DatasetFSReader._safe_read_from_path(
                fs=fs,
                dataset_path=str(Path(tmpdir) / "test-dataset"),
                data_type="documents",
            )

            # Verify order
            assert len(result_df) == 3
            assert result_df.iloc[0]["id"] == "id_0"
            assert result_df.iloc[1]["id"] == "id_1"
            assert result_df.iloc[2]["id"] == "id_2"

    def test_safe_read_from_path_with_parallel_workers(self):
        """Test that parallel workers produce correct results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dataset directory structure
            docs_dir = Path(tmpdir) / "test-dataset" / "documents"
            docs_dir.mkdir(parents=True)

            # Create multiple parquet files
            expected_ids = []
            for i in range(5):
                df = pd.DataFrame(
                    {
                        "id": [f"id_{i}"],
                        "values": [[float(i)]],
                        "sparse_values": [None],
                        "metadata": [None],
                        "blob": [None],
                    }
                )
                expected_ids.append(f"id_{i}")
                path = docs_dir / f"{i:04d}.parquet"
                table = pa.Table.from_pandas(df)
                pq.write_table(table, str(path))

            from fsspec.implementations.local import LocalFileSystem

            fs = LocalFileSystem()

            # Read with parallel processing enabled
            cfg.Cache.max_parallel_downloads = 3
            result_df = DatasetFSReader._safe_read_from_path(
                fs=fs,
                dataset_path=str(Path(tmpdir) / "test-dataset"),
                data_type="documents",
            )

            # Verify all data is present and in correct order
            assert len(result_df) == 5
            result_ids = result_df["id"].tolist()
            assert result_ids == expected_ids

    def test_safe_read_from_path_single_file_uses_serial(self):
        """Test that single file datasets use serial processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir) / "test-dataset" / "documents"
            docs_dir.mkdir(parents=True)

            # Create single parquet file
            df = pd.DataFrame(
                {
                    "id": ["id_0"],
                    "values": [[0.1]],
                    "sparse_values": [None],
                    "metadata": [None],
                    "blob": [None],
                }
            )
            path = docs_dir / "0000.parquet"
            table = pa.Table.from_pandas(df)
            pq.write_table(table, str(path))

            from fsspec.implementations.local import LocalFileSystem

            fs = LocalFileSystem()

            # Even with high max_workers, should use serial for single file
            cfg.Cache.max_parallel_downloads = 10
            result_df = DatasetFSReader._safe_read_from_path(
                fs=fs,
                dataset_path=str(Path(tmpdir) / "test-dataset"),
                data_type="documents",
            )

            assert len(result_df) == 1
            assert result_df.iloc[0]["id"] == "id_0"

    def test_safe_read_from_path_respects_max_workers_limit(self):
        """Test that max_workers limit is respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir) / "test-dataset" / "documents"
            docs_dir.mkdir(parents=True)

            # Create 10 parquet files
            for i in range(10):
                df = pd.DataFrame(
                    {
                        "id": [f"id_{i}"],
                        "values": [[float(i)]],
                        "sparse_values": [None],
                        "metadata": [None],
                        "blob": [None],
                    }
                )
                path = docs_dir / f"{i:04d}.parquet"
                table = pa.Table.from_pandas(df)
                pq.write_table(table, str(path))

            from fsspec.implementations.local import LocalFileSystem

            fs = LocalFileSystem()

            # Set low max_workers
            cfg.Cache.max_parallel_downloads = 2

            result_df = DatasetFSReader._safe_read_from_path(
                fs=fs,
                dataset_path=str(Path(tmpdir) / "test-dataset"),
                data_type="documents",
            )

            # Verify all files were processed
            assert len(result_df) == 10
            # Verify order is maintained
            for i in range(10):
                assert result_df.iloc[i]["id"] == f"id_{i}"
