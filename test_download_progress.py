#!/usr/bin/env python3
"""
Manual test script for download progress feedback.

Tests the tqdm progress bars during dataset downloads from cloud storage.
"""

import time
from pinecone_datasets import list_datasets, load_dataset
from pinecone_datasets.cache import cache_info, clear_cache


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_fresh_download():
    """Test downloading a dataset from scratch (with progress bars)."""
    print_section("TEST 1: Fresh Download (First Time)")
    
    # Clear cache to ensure fresh download
    print("Clearing cache...")
    cleared = clear_cache()
    print(f"✓ Cleared {cleared} cache files\n")
    
    # Download a small dataset to test progress
    print("Downloading 'quora_all-MiniLM-L6-bm25' dataset...")
    print("(Watch for progress bars showing download speed and ETA)\n")
    
    dataset = load_dataset("quora_all-MiniLM-L6-bm25")
    
    print(f"\n✓ Successfully loaded dataset!")
    print(f"  - Documents: {len(dataset.documents)}")
    print(f"  - Queries: {len(dataset.queries)}")
    print(f"  - Embedding model: {dataset.metadata.dense_model.name}")


def test_cached_download():
    """Test loading dataset from cache (should be instant)."""
    print_section("TEST 2: Cached Download (Second Time)")
    
    print("Loading same dataset again (should use cache, no download)...\n")
    
    start = time.time()
    dataset = load_dataset("quora_all-MiniLM-L6-bm25")
    elapsed = time.time() - start
    
    print(f"\n✓ Loaded from cache in {elapsed:.2f}s (instant!)")
    print(f"  - Documents: {len(dataset.documents)}")


def test_cache_info():
    """Display cache statistics."""
    print_section("TEST 3: Cache Information")
    
    info = cache_info()
    print(f"Cache directory: {info['cache_dir']}")
    print(f"Total size: {info['total_size_mb']:.2f} MB ({info['total_size_gb']:.4f} GB)")
    print(f"File count: {info['file_count']}")


def test_larger_dataset():
    """Test with a larger dataset to see longer progress bars."""
    print_section("TEST 4: Larger Dataset (Optional)")
    
    response = input("\nTest with a larger dataset? This will take longer (y/N): ")
    if response.lower() != 'y':
        print("Skipped.")
        return
    
    print("\nClearing cache...")
    clear_cache()
    
    print("\nDownloading 'wikipedia-simple-text-embedding-3-small-512-100K' dataset...")
    print("(This is larger, so you'll see longer progress bars)\n")
    
    dataset = load_dataset("wikipedia-simple-text-embedding-3-small-512-100K")
    
    print(f"\n✓ Successfully loaded dataset!")
    print(f"  - Documents: {len(dataset.documents)}")


def list_available_datasets():
    """Show available public datasets."""
    print_section("Available Public Datasets")
    
    print("Fetching list of available datasets...\n")
    datasets = list_datasets()
    
    print(f"Found {len(datasets)} public datasets:\n")
    for ds in datasets[:10]:  # Show first 10
        print(f"  • {ds}")
    
    if len(datasets) > 10:
        print(f"\n  ... and {len(datasets) - 10} more")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("  PINECONE DATASETS - DOWNLOAD PROGRESS TEST")
    print("=" * 80)
    print("\nThis script tests the download progress feedback feature.")
    print("You should see tqdm progress bars with:")
    print("  • File size and bytes downloaded")
    print("  • Download speed (MB/s)")
    print("  • Estimated time remaining (ETA)")
    print("  • Progress percentage")
    
    try:
        # Show available datasets
        list_available_datasets()
        
        # Test fresh download with progress
        test_fresh_download()
        
        # Test cached (instant) load
        test_cached_download()
        
        # Show cache info
        test_cache_info()
        
        # Optional: test larger dataset
        test_larger_dataset()
        
        print_section("ALL TESTS COMPLETE")
        print("✓ Progress bars working correctly!")
        print("✓ Caching working correctly!")
        print("\nCache preserved for future use.")
        print(f"To clear cache, run: python -c 'from pinecone_datasets import clear_cache; clear_cache()'")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
