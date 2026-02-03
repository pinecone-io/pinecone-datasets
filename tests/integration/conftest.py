import tempfile

import pytest

from pinecone_datasets import clear_cache, set_cache_dir


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
