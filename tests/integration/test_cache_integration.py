import logging
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal as pd_assert_frame_equal

from pinecone_datasets import Catalog, Dataset, DatasetMetadata, DenseModelMetadata, set_cache_dir, cache_info, clear_cache
from pinecone_datasets.cache import CacheManager
from pinecone_datasets.fs import get_cloud_fs, get_cached_path, should_use_cache

logger = logging.getLogger(__name__)


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(autouse=True)
def setup_cache_dir(temp_cache_dir):
    """Set up cache directory for each test."""
    set_cache_dir(temp_cache_dir)
    yield
    # Cleanup after test
    clear_cache()


class TestCacheIntegration:
    def test_cache_manager_with_local_fs(self, temp_cache_dir, tmpdir):
        """Test that caching works with local filesystem."""
        # Create a local test file
        test_file = tmpdir.join("test.parquet")
        test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        test_data.to_parquet(str(test_file))
        
        fs = get_cloud_fs(str(tmpdir))
        cache_manager = CacheManager(cache_dir=temp_cache_dir)
        
        # Get cached path
        cached_path = cache_manager.get_cached_path(str(test_file), fs)
        
        assert os.path.exists(cached_path)
        assert cached_path.startswith(temp_cache_dir)
        
        # Verify content is the same
        cached_data = pd.read_parquet(cached_path)
        pd_assert_frame_equal(cached_data, test_data)

    def test_cache_hit_avoids_redownload(self, temp_cache_dir, tmpdir):
        """Test that cached files are reused without re-downloading."""
        # Create a local test file
        test_file = tmpdir.join("test.parquet")
        test_data = pd.DataFrame({"col1": [1, 2, 3]})
        test_data.to_parquet(str(test_file))
        
        fs = get_cloud_fs(str(tmpdir))
        cache_manager = CacheManager(cache_dir=temp_cache_dir)
        
        # First access - should download
        cached_path_1 = cache_manager.get_cached_path(str(test_file), fs)
        first_mtime = os.path.getmtime(cached_path_1)
        
        # Small delay to ensure mtime would change if file was rewritten
        time.sleep(0.1)
        
        # Second access - should use cache
        cached_path_2 = cache_manager.get_cached_path(str(test_file), fs)
        second_mtime = os.path.getmtime(cached_path_2)
        
        assert cached_path_1 == cached_path_2
        assert first_mtime == second_mtime  # File was not rewritten

    def test_cache_invalidation_on_size_change(self, temp_cache_dir, tmpdir):
        """Test that cache is invalidated when remote file size changes."""
        test_file = tmpdir.join("test.parquet")
        
        # Create initial file
        test_data_1 = pd.DataFrame({"col1": [1, 2, 3]})
        test_data_1.to_parquet(str(test_file))
        
        fs = get_cloud_fs(str(tmpdir))
        cache_manager = CacheManager(cache_dir=temp_cache_dir)
        
        # First access
        cached_path = cache_manager.get_cached_path(str(test_file), fs)
        cached_data_1 = pd.read_parquet(cached_path)
        pd_assert_frame_equal(cached_data_1, test_data_1)
        
        # Modify remote file (different size)
        test_data_2 = pd.DataFrame({"col1": [1, 2, 3, 4, 5, 6]})
        test_data_2.to_parquet(str(test_file))
        
        # Second access - should re-download due to size change
        cached_path_2 = cache_manager.get_cached_path(str(test_file), fs)
        cached_data_2 = pd.read_parquet(cached_path_2)
        pd_assert_frame_equal(cached_data_2, test_data_2)

    def test_should_use_cache_logic(self):
        """Test that should_use_cache returns correct values."""
        # Cloud paths should use cache by default (if enabled in config)
        from pinecone_datasets import cfg
        original_use_cache = cfg.Cache.use_cache
        
        try:
            cfg.Cache.use_cache = True
            assert should_use_cache("gs://bucket/file.parquet") is True
            assert should_use_cache("s3://bucket/file.parquet") is True
            
            # Local paths should not use cache by default
            assert should_use_cache("/local/file.parquet") is False
            
            # Explicit override
            assert should_use_cache("gs://bucket/file.parquet", use_cache=False) is False
            assert should_use_cache("/local/file.parquet", use_cache=True) is True
            
            # When config disabled
            cfg.Cache.use_cache = False
            assert should_use_cache("gs://bucket/file.parquet") is False
        finally:
            cfg.Cache.use_cache = original_use_cache

    def test_get_cached_path_for_local_file(self, tmpdir):
        """Test that get_cached_path returns original path for local files (no caching)."""
        test_file = tmpdir.join("test.parquet")
        test_data = pd.DataFrame({"col1": [1, 2, 3]})
        test_data.to_parquet(str(test_file))
        
        fs = get_cloud_fs(str(tmpdir))
        
        # For local files, should return original path (no caching)
        result_path = get_cached_path(str(test_file), fs, use_cache=False)
        assert result_path == str(test_file)

    def test_cache_info_and_clear(self, temp_cache_dir, tmpdir):
        """Test cache_info and clear_cache functions."""
        # Initially empty
        info = cache_info()
        assert info["file_count"] == 0
        assert info["total_size_bytes"] == 0
        
        # Create and cache a file
        test_file = tmpdir.join("test.parquet")
        test_data = pd.DataFrame({"col1": list(range(1000))})
        test_data.to_parquet(str(test_file))
        
        fs = get_cloud_fs(str(tmpdir))
        cache_manager = CacheManager(cache_dir=temp_cache_dir)
        cache_manager.get_cached_path(str(test_file), fs)
        
        # Check cache info
        info = cache_info()
        assert info["file_count"] == 1
        assert info["total_size_bytes"] > 0
        
        # Clear cache
        count = clear_cache()
        assert count >= 1
        
        # Verify cache is empty
        info = cache_info()
        assert info["file_count"] == 0

    def test_cache_with_multiple_files(self, temp_cache_dir, tmpdir):
        """Test caching multiple files."""
        cache_manager = CacheManager(cache_dir=temp_cache_dir)
        fs = get_cloud_fs(str(tmpdir))
        
        # Create multiple test files
        files = []
        for i in range(3):
            test_file = tmpdir.join(f"test_{i}.parquet")
            test_data = pd.DataFrame({"col1": [i, i+1, i+2]})
            test_data.to_parquet(str(test_file))
            files.append(str(test_file))
        
        # Cache all files
        cached_paths = []
        for file_path in files:
            cached_path = cache_manager.get_cached_path(file_path, fs)
            cached_paths.append(cached_path)
            assert os.path.exists(cached_path)
        
        # Verify all are cached
        for file_path, cached_path in zip(files, cached_paths):
            assert cache_manager.is_cached(file_path, fs)
            assert os.path.exists(cached_path)
        
        # Verify cache info
        info = cache_manager.get_cache_info()
        assert info["file_count"] == 3

    def test_cache_directory_creation(self):
        """Test that cache directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "new_cache", "subdir")
            assert not os.path.exists(cache_dir)
            
            cache_manager = CacheManager(cache_dir=cache_dir)
            assert os.path.exists(cache_dir)

    @pytest.mark.slow
    def test_load_public_dataset_with_cache(self, temp_cache_dir):
        """Integration test: Load a small public dataset with caching enabled."""
        # Use a small dataset for faster testing
        catalog = Catalog()
        
        # First load - downloads and caches
        ds1 = catalog.load_dataset("langchain-python-docs-text-embedding-ada-002")
        assert ds1 is not None
        assert len(ds1.documents) > 0
        
        # Check that cache has files
        info = cache_info()
        initial_file_count = info["file_count"]
        assert initial_file_count > 0
        
        # Second load - should use cache (faster)
        start_time = time.time()
        ds2 = catalog.load_dataset("langchain-python-docs-text-embedding-ada-002")
        cache_load_time = time.time() - start_time
        
        assert ds2 is not None
        # Should have same number of documents
        assert len(ds2.documents) == len(ds1.documents)
        
        logger.info(f"Cached load time: {cache_load_time:.2f}s")

    def test_clear_cache_with_pattern(self, temp_cache_dir, tmpdir):
        """Test clearing cache with specific pattern."""
        cache_manager = CacheManager(cache_dir=temp_cache_dir)
        fs = get_cloud_fs(str(tmpdir))
        
        # Create files with different extensions
        parquet_file = tmpdir.join("test.parquet")
        pd.DataFrame({"col1": [1, 2, 3]}).to_parquet(str(parquet_file))
        
        # Create a non-parquet file in cache
        Path(temp_cache_dir, "other.txt").write_text("some text")
        
        # Cache the parquet file
        cache_manager.get_cached_path(str(parquet_file), fs)
        
        # Clear only parquet files
        count = cache_manager.clear_cache(pattern="*.parquet")
        assert count >= 1
        
        # Verify txt file still exists
        assert Path(temp_cache_dir, "other.txt").exists()
