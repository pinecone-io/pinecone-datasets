"""Tests for parallel downloading functionality."""

import os
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from pinecone_datasets.dataset_fsreader import DatasetFSReader


class TestParallelDownloads:
    """Test parallel downloading of parquet files."""

    def test_parallel_download_multiple_files(self):
        """Test that multiple files are downloaded in parallel"""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        # Simulate 5 parquet files
        mock_fs.glob.return_value = [
            "gs://bucket/dataset/documents/part-0.parquet",
            "gs://bucket/dataset/documents/part-1.parquet",
            "gs://bucket/dataset/documents/part-2.parquet",
            "gs://bucket/dataset/documents/part-3.parquet",
            "gs://bucket/dataset/documents/part-4.parquet",
        ]

        # Create mock dataframes
        mock_df = pd.DataFrame(
            {"id": ["1"], "values": [[0.1]], "sparse_values": [None], "metadata": [{}]}
        )

        with patch(
            "pinecone_datasets.dataset_fsreader.DatasetFSReader._download_and_read_parquet",
            return_value=mock_df,
        ) as mock_download:
            # Set max_parallel_downloads to 4 for testing
            with patch("pinecone_datasets.cfg.Cache.max_parallel_downloads", 4):
                df = DatasetFSReader._safe_read_from_path(
                    mock_fs, "gs://bucket/dataset", "documents"
                )

                # Verify all files were processed
                assert len(df) == 5  # 5 files * 1 row each
                assert mock_download.call_count == 5

    def test_parallel_download_single_file_uses_serial(self):
        """Test that single file doesn't use parallel executor"""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        # Only one file
        mock_fs.glob.return_value = ["gs://bucket/dataset/documents/part-0.parquet"]

        mock_df = pd.DataFrame(
            {"id": ["1"], "values": [[0.1]], "sparse_values": [None], "metadata": [{}]}
        )

        with patch(
            "pinecone_datasets.dataset_fsreader.DatasetFSReader._download_and_read_parquet",
            return_value=mock_df,
        ):
            df = DatasetFSReader._safe_read_from_path(
                mock_fs, "gs://bucket/dataset", "documents"
            )

            # Should still work correctly with single file
            assert len(df) == 1

    def test_parallel_download_max_workers_1_uses_serial(self):
        """Test that max_workers=1 forces serial execution"""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_fs.glob.return_value = [
            "gs://bucket/dataset/documents/part-0.parquet",
            "gs://bucket/dataset/documents/part-1.parquet",
        ]

        mock_df = pd.DataFrame(
            {"id": ["1"], "values": [[0.1]], "sparse_values": [None], "metadata": [{}]}
        )

        with patch(
            "pinecone_datasets.dataset_fsreader.DatasetFSReader._download_and_read_parquet",
            return_value=mock_df,
        ):
            # Force serial execution with max_workers=1
            with patch("pinecone_datasets.cfg.Cache.max_parallel_downloads", 1):
                df = DatasetFSReader._safe_read_from_path(
                    mock_fs, "gs://bucket/dataset", "documents"
                )

                # Should work correctly in serial mode
                assert len(df) == 2

    def test_parallel_download_error_handling(self):
        """Test that errors in parallel downloads are properly raised"""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_fs.glob.return_value = [
            "gs://bucket/dataset/documents/part-0.parquet",
            "gs://bucket/dataset/documents/part-1.parquet",
        ]

        def side_effect_error(*args, **kwargs):
            raise OSError("Download failed")

        with patch(
            "pinecone_datasets.dataset_fsreader.DatasetFSReader._download_and_read_parquet",
            side_effect=side_effect_error,
        ):
            with patch("pinecone_datasets.cfg.Cache.max_parallel_downloads", 2):
                with pytest.raises(OSError, match="Download failed"):
                    DatasetFSReader._safe_read_from_path(
                        mock_fs, "gs://bucket/dataset", "documents"
                    )

    def test_parallel_download_limits_workers_to_file_count(self):
        """Test that workers don't exceed number of files"""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        # Only 2 files but max_workers is 10
        mock_fs.glob.return_value = [
            "gs://bucket/dataset/documents/part-0.parquet",
            "gs://bucket/dataset/documents/part-1.parquet",
        ]

        mock_df = pd.DataFrame(
            {"id": ["1"], "values": [[0.1]], "sparse_values": [None], "metadata": [{}]}
        )

        with patch(
            "pinecone_datasets.dataset_fsreader.DatasetFSReader._download_and_read_parquet",
            return_value=mock_df,
        ):
            with patch("pinecone_datasets.cfg.Cache.max_parallel_downloads", 10):
                df = DatasetFSReader._safe_read_from_path(
                    mock_fs, "gs://bucket/dataset", "documents"
                )

                # Should work correctly with fewer files than workers
                assert len(df) == 2

    def test_download_and_read_parquet_with_cache(self):
        """Test _download_and_read_parquet helper function with caching"""
        mock_fs = Mock()
        path = "bucket/dataset/documents/part-0.parquet"
        protocol = "gs://"

        mock_df = pd.DataFrame(
            {"id": ["1"], "values": [[0.1]], "sparse_values": [None], "metadata": [{}]}
        )

        with patch("pinecone_datasets.dataset_fsreader.get_cached_path", return_value="/tmp/cached.parquet"):
            with patch("pyarrow.parquet.read_pandas") as mock_read:
                mock_piece = Mock()
                mock_piece.to_pandas.return_value = mock_df
                mock_read.return_value = mock_piece

                df = DatasetFSReader._download_and_read_parquet(
                    path, use_cache=True, protocol=protocol, fs=mock_fs
                )

                assert len(df) == 1
                assert df["id"][0] == "1"

    def test_download_and_read_parquet_without_cache(self):
        """Test _download_and_read_parquet helper function without caching"""
        mock_fs = Mock()
        path = "/local/dataset/documents/part-0.parquet"

        mock_df = pd.DataFrame(
            {"id": ["1"], "values": [[0.1]], "sparse_values": [None], "metadata": [{}]}
        )

        with patch("pyarrow.parquet.read_pandas") as mock_read:
            mock_piece = Mock()
            mock_piece.to_pandas.return_value = mock_df
            mock_read.return_value = mock_piece

            df = DatasetFSReader._download_and_read_parquet(
                path, use_cache=False, protocol=None, fs=mock_fs
            )

            assert len(df) == 1
            assert df["id"][0] == "1"
            # Verify read_pandas was called with filesystem parameter
            mock_read.assert_called_once_with(path, filesystem=mock_fs)
