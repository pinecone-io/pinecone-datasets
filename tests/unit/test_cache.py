import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from pinecone_datasets.cache import CacheManager, cache_info, clear_cache, set_cache_dir


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def cache_manager(temp_cache_dir):
    """Create a CacheManager instance with temporary cache directory."""
    return CacheManager(cache_dir=temp_cache_dir)


@pytest.fixture
def mock_fs():
    """Create a mock filesystem object."""
    fs = MagicMock()
    fs.size.return_value = 1024 * 1024  # 1 MB
    fs.open.return_value.__enter__ = Mock(return_value=Mock())
    fs.open.return_value.__exit__ = Mock(return_value=False)
    return fs


class TestCacheManager:
    def test_init_creates_cache_dir(self, temp_cache_dir):
        """Test that CacheManager creates cache directory on initialization."""
        cache_dir = os.path.join(temp_cache_dir, "new_cache")
        assert not os.path.exists(cache_dir)

        manager = CacheManager(cache_dir=cache_dir)
        assert os.path.exists(cache_dir)
        assert manager.cache_dir == cache_dir

    def test_get_cache_path_deterministic(self, cache_manager):
        """Test that cache path generation is deterministic."""
        url = "gs://bucket/file.parquet"
        path1 = cache_manager._get_cache_path(url)
        path2 = cache_manager._get_cache_path(url)
        assert path1 == path2

    def test_get_cache_path_preserves_extension(self, cache_manager):
        """Test that file extension is preserved in cache path."""
        url = "gs://bucket/file.parquet"
        path = cache_manager._get_cache_path(url)
        assert path.endswith(".parquet")

    def test_get_cache_path_different_urls(self, cache_manager):
        """Test that different URLs produce different cache paths."""
        url1 = "gs://bucket/file1.parquet"
        url2 = "gs://bucket/file2.parquet"
        path1 = cache_manager._get_cache_path(url1)
        path2 = cache_manager._get_cache_path(url2)
        assert path1 != path2

    def test_write_and_read_metadata(self, cache_manager, temp_cache_dir):
        """Test metadata writing and reading."""
        metadata_path = os.path.join(temp_cache_dir, "test.meta")
        remote_url = "gs://bucket/file.parquet"
        expected_size = 1024
        downloaded_bytes = 512

        cache_manager._write_metadata(
            metadata_path, remote_url, expected_size, downloaded_bytes
        )
        assert os.path.exists(metadata_path)

        metadata = cache_manager._read_metadata(metadata_path)
        assert metadata["remote_url"] == remote_url
        assert metadata["expected_size"] == expected_size
        assert metadata["downloaded_bytes"] == downloaded_bytes

    def test_read_metadata_invalid_file(self, cache_manager, temp_cache_dir):
        """Test that reading invalid metadata returns None."""
        # Non-existent file
        metadata = cache_manager._read_metadata(
            os.path.join(temp_cache_dir, "nonexistent.meta")
        )
        assert metadata is None

        # Invalid JSON
        invalid_path = os.path.join(temp_cache_dir, "invalid.meta")
        with open(invalid_path, "w") as f:
            f.write("not json")
        metadata = cache_manager._read_metadata(invalid_path)
        assert metadata is None

    def test_validate_cache_valid(self, cache_manager, mock_fs, temp_cache_dir):
        """Test cache validation for valid cached file."""
        cache_path = os.path.join(temp_cache_dir, "cached.parquet")
        remote_url = "gs://bucket/file.parquet"

        # Create cached file with correct size
        file_size = 1024
        with open(cache_path, "wb") as f:
            f.write(b"x" * file_size)

        mock_fs.size.return_value = file_size
        assert cache_manager._validate_cache(cache_path, remote_url, mock_fs) is True

    def test_validate_cache_size_mismatch(self, cache_manager, mock_fs, temp_cache_dir):
        """Test cache validation fails when size doesn't match."""
        cache_path = os.path.join(temp_cache_dir, "cached.parquet")
        remote_url = "gs://bucket/file.parquet"

        # Create cached file with incorrect size
        with open(cache_path, "wb") as f:
            f.write(b"x" * 512)

        mock_fs.size.return_value = 1024
        assert cache_manager._validate_cache(cache_path, remote_url, mock_fs) is False

    def test_validate_cache_nonexistent(self, cache_manager, mock_fs):
        """Test cache validation fails for non-existent file."""
        assert (
            cache_manager._validate_cache(
                "/nonexistent/file", "gs://bucket/file", mock_fs
            )
            is False
        )

    def test_validate_partial_valid(self, cache_manager, mock_fs, temp_cache_dir):
        """Test partial download validation for valid metadata."""
        metadata_path = os.path.join(temp_cache_dir, "test.meta")
        remote_url = "gs://bucket/file.parquet"
        expected_size = 1024

        cache_manager._write_metadata(metadata_path, remote_url, expected_size, 512)
        mock_fs.size.return_value = expected_size

        assert (
            cache_manager._validate_partial(metadata_path, remote_url, mock_fs) is True
        )

    def test_validate_partial_url_mismatch(
        self, cache_manager, mock_fs, temp_cache_dir
    ):
        """Test partial validation fails when URL doesn't match."""
        metadata_path = os.path.join(temp_cache_dir, "test.meta")
        remote_url = "gs://bucket/file.parquet"
        different_url = "gs://bucket/other.parquet"

        cache_manager._write_metadata(metadata_path, remote_url, 1024, 512)
        mock_fs.size.return_value = 1024

        assert (
            cache_manager._validate_partial(metadata_path, different_url, mock_fs)
            is False
        )

    def test_validate_partial_size_changed(
        self, cache_manager, mock_fs, temp_cache_dir
    ):
        """Test partial validation fails when remote file size changed."""
        metadata_path = os.path.join(temp_cache_dir, "test.meta")
        remote_url = "gs://bucket/file.parquet"

        cache_manager._write_metadata(metadata_path, remote_url, 1024, 512)
        mock_fs.size.return_value = 2048  # Remote file changed size

        assert (
            cache_manager._validate_partial(metadata_path, remote_url, mock_fs) is False
        )

    def test_download_file(self, cache_manager, temp_cache_dir):
        """Test downloading a file from remote storage."""
        remote_url = "gs://bucket/file.parquet"
        output_path = os.path.join(temp_cache_dir, "output.parquet")

        # Mock filesystem with file content
        mock_fs = MagicMock()
        test_data = b"test file content" * 1000
        mock_fs.size.return_value = len(test_data)

        mock_file = MagicMock()
        mock_file.read.side_effect = [test_data, b""]  # Return data then EOF
        mock_fs.open.return_value.__enter__.return_value = mock_file
        mock_fs.open.return_value.__exit__.return_value = False

        cache_manager._download_file(remote_url, mock_fs, output_path, start_byte=0)

        assert os.path.exists(output_path)
        with open(output_path, "rb") as f:
            assert f.read() == test_data

    def test_download_file_resume(self, cache_manager, temp_cache_dir):
        """Test resuming a partial download."""
        remote_url = "gs://bucket/file.parquet"
        output_path = os.path.join(temp_cache_dir, "output.parquet")

        # Create partial file
        partial_data = b"already downloaded"
        with open(output_path, "wb") as f:
            f.write(partial_data)

        # Mock filesystem
        mock_fs = MagicMock()
        remaining_data = b"remaining content"
        total_size = len(partial_data) + len(remaining_data)
        mock_fs.size.return_value = total_size

        mock_file = MagicMock()
        mock_file.read.side_effect = [remaining_data, b""]
        mock_fs.open.return_value.__enter__.return_value = mock_file
        mock_fs.open.return_value.__exit__.return_value = False

        cache_manager._download_file(
            remote_url, mock_fs, output_path, start_byte=len(partial_data)
        )

        with open(output_path, "rb") as f:
            content = f.read()
            assert content == partial_data + remaining_data

    def test_is_cached_true(self, cache_manager, mock_fs, temp_cache_dir):
        """Test is_cached returns True for valid cached file."""
        remote_url = "gs://bucket/file.parquet"
        cache_path = cache_manager._get_cache_path(remote_url)

        # Create valid cached file
        file_size = 1024
        with open(cache_path, "wb") as f:
            f.write(b"x" * file_size)

        mock_fs.size.return_value = file_size
        assert cache_manager.is_cached(remote_url, mock_fs) is True

    def test_is_cached_false(self, cache_manager, mock_fs):
        """Test is_cached returns False for non-cached file."""
        remote_url = "gs://bucket/file.parquet"
        assert cache_manager.is_cached(remote_url, mock_fs) is False

    def test_clear_cache_all(self, cache_manager, temp_cache_dir):
        """Test clearing all cache files."""
        # Create some cache files
        Path(temp_cache_dir, "file1.parquet").write_text("data1")
        Path(temp_cache_dir, "file2.parquet").write_text("data2")
        Path(temp_cache_dir, "file1.parquet.meta").write_text("{}")

        count = cache_manager.clear_cache()
        assert count == 3
        assert len(list(Path(temp_cache_dir).glob("*"))) == 0

    def test_clear_cache_pattern(self, cache_manager, temp_cache_dir):
        """Test clearing cache files matching pattern."""
        # Create files with different extensions
        Path(temp_cache_dir, "file1.parquet").write_text("data1")
        Path(temp_cache_dir, "file2.csv").write_text("data2")

        count = cache_manager.clear_cache(pattern="*.parquet")
        assert count == 1
        assert not Path(temp_cache_dir, "file1.parquet").exists()
        assert Path(temp_cache_dir, "file2.csv").exists()

    def test_get_cache_info_empty(self, cache_manager):
        """Test cache info for empty cache."""
        info = cache_manager.get_cache_info()
        assert info["file_count"] == 0
        assert info["total_size_bytes"] == 0
        assert info["cache_dir"] == cache_manager.cache_dir

    def test_get_cache_info_with_files(self, cache_manager, temp_cache_dir):
        """Test cache info with files in cache."""
        # Create test files
        Path(temp_cache_dir, "file1.parquet").write_bytes(b"x" * 1024)
        Path(temp_cache_dir, "file2.parquet").write_bytes(b"x" * 2048)
        Path(temp_cache_dir, "file1.parquet.meta").write_text("{}")  # Should be ignored

        info = cache_manager.get_cache_info()
        assert info["file_count"] == 2
        assert info["total_size_bytes"] == 1024 + 2048
        assert info["total_size_mb"] == round((1024 + 2048) / (1024 * 1024), 2)

    def test_get_cached_path_new_download(self, cache_manager, temp_cache_dir):
        """Test getting cached path for new file (triggers download)."""
        remote_url = "gs://bucket/file.parquet"

        # Mock filesystem
        mock_fs = MagicMock()
        test_data = b"test content"
        mock_fs.size.return_value = len(test_data)

        mock_file = MagicMock()
        mock_file.read.side_effect = [test_data, b""]
        mock_fs.open.return_value.__enter__.return_value = mock_file
        mock_fs.open.return_value.__exit__.return_value = False

        cached_path = cache_manager.get_cached_path(remote_url, mock_fs)

        assert os.path.exists(cached_path)
        assert cached_path.startswith(temp_cache_dir)
        with open(cached_path, "rb") as f:
            assert f.read() == test_data

    def test_get_cached_path_already_cached(self, cache_manager, temp_cache_dir):
        """Test getting cached path when file is already cached."""
        remote_url = "gs://bucket/file.parquet"
        cache_path = cache_manager._get_cache_path(remote_url)

        # Pre-populate cache
        test_data = b"cached content"
        with open(cache_path, "wb") as f:
            f.write(test_data)

        mock_fs = MagicMock()
        mock_fs.size.return_value = len(test_data)

        # Should return cached path without downloading
        cached_path = cache_manager.get_cached_path(remote_url, mock_fs)
        assert cached_path == cache_path
        mock_fs.open.assert_not_called()  # No download occurred


class TestCacheGlobalFunctions:
    def test_set_cache_dir(self, temp_cache_dir):
        """Test setting global cache directory."""
        set_cache_dir(temp_cache_dir)
        info = cache_info()
        assert info["cache_dir"] == temp_cache_dir

    def test_cache_info(self, temp_cache_dir):
        """Test global cache_info function."""
        set_cache_dir(temp_cache_dir)
        Path(temp_cache_dir, "test.parquet").write_bytes(b"x" * 1024)

        info = cache_info()
        assert info["file_count"] == 1
        assert info["total_size_bytes"] == 1024

    def test_clear_cache_global(self, temp_cache_dir):
        """Test global clear_cache function."""
        set_cache_dir(temp_cache_dir)
        Path(temp_cache_dir, "test.parquet").write_text("data")

        count = clear_cache()
        assert count == 1
        assert len(list(Path(temp_cache_dir).glob("*"))) == 0
