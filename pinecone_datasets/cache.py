import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

from .fs import CloudOrLocalFS
from .tqdm import tqdm

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages local caching of remote dataset files with support for resumable downloads.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the CacheManager.

        Args:
            cache_dir: Directory to store cached files. Defaults to ~/.pinecone-datasets/cache
        """
        from . import cfg

        self.cache_dir = cache_dir or cfg.Cache.cache_dir
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, remote_url: str) -> str:
        """
        Generate deterministic cache path from remote URL.

        Args:
            remote_url: Remote file URL

        Returns:
            Local cache path with preserved file extension
        """
        # Hash the URL for a deterministic cache path
        url_hash = hashlib.sha256(remote_url.encode()).hexdigest()[:16]
        # Preserve file extension for proper handling (e.g., .parquet)
        ext = os.path.splitext(remote_url)[1]
        cache_filename = f"{url_hash}{ext}"
        return os.path.join(self.cache_dir, cache_filename)

    def _get_metadata_path(self, cache_path: str) -> str:
        """Get metadata file path for a cache file."""
        return cache_path + ".meta"

    def _get_partial_path(self, cache_path: str) -> str:
        """Get partial download file path."""
        return cache_path + ".partial"

    def _get_file_etag(self, remote_url: str, fs: CloudOrLocalFS) -> Optional[str]:
        """
        Get ETag or modification time for file content validation.

        Args:
            remote_url: Remote file URL
            fs: Filesystem object

        Returns:
            ETag string if available, None otherwise
        """
        try:
            info = fs.info(remote_url)
            # Try to get ETag first (most reliable for content changes)
            return info.get("ETag") or info.get("etag") or info.get("mtime")
        except Exception:
            return None

    def _write_metadata(
        self,
        metadata_path: str,
        remote_url: str,
        expected_size: int,
        downloaded_bytes: int,
        etag: Optional[str] = None,
    ) -> None:
        """
        Write metadata for a partial download.

        Args:
            metadata_path: Path to metadata file
            remote_url: Remote file URL
            expected_size: Expected total file size
            downloaded_bytes: Number of bytes downloaded so far
            etag: ETag or modification time for content validation
        """
        metadata = {
            "remote_url": remote_url,
            "expected_size": expected_size,
            "downloaded_bytes": downloaded_bytes,
            "etag": etag,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

    def _read_metadata(self, metadata_path: str) -> Optional[dict]:
        """
        Read metadata from a partial download.

        Returns:
            Metadata dict or None if metadata is invalid
        """
        try:
            with open(metadata_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, OSError, ValueError):
            return None

    def _validate_cache(
        self, cache_path: str, remote_url: str, fs: CloudOrLocalFS
    ) -> bool:
        """
        Validate that a cached file is complete and matches the remote file.

        Args:
            cache_path: Local cache path
            remote_url: Remote file URL
            fs: Filesystem object

        Returns:
            True if cache is valid, False otherwise
        """
        try:
            if not os.path.exists(cache_path):
                return False

            local_size = os.path.getsize(cache_path)
            remote_size = fs.size(remote_url)

            return local_size == remote_size
        except Exception as e:
            logger.debug(f"Cache validation failed: {e}")
            return False

    def _validate_partial(
        self, metadata_path: str, remote_url: str, fs: CloudOrLocalFS
    ) -> bool:
        """
        Validate that partial download metadata is still valid for resuming.

        Args:
            metadata_path: Path to metadata file
            remote_url: Remote file URL
            fs: Filesystem object

        Returns:
            True if partial download can be resumed, False otherwise
        """
        metadata = self._read_metadata(metadata_path)
        if not metadata:
            return False

        try:
            # Check URL matches
            if metadata["remote_url"] != remote_url:
                return False

            # Check remote file size matches
            remote_size = fs.size(remote_url)
            if metadata["expected_size"] != remote_size:
                logger.debug("Remote file size changed, cannot resume download")
                return False

            # Check ETag/mtime if available to detect content changes
            if "etag" in metadata and metadata["etag"]:
                current_etag = self._get_file_etag(remote_url, fs)
                if current_etag and current_etag != metadata["etag"]:
                    logger.debug(
                        "Remote file content changed (ETag mismatch), cannot resume download"
                    )
                    return False

            return True
        except Exception as e:
            logger.debug(f"Partial validation failed: {e}")
            return False

    def _download_file(
        self,
        remote_url: str,
        fs: CloudOrLocalFS,
        output_path: str,
        start_byte: int = 0,
        etag: Optional[str] = None,
        show_progress: bool = True,
    ) -> None:
        """
        Download a file from remote storage with resume support and progress feedback.

        Args:
            remote_url: Remote file URL
            fs: Filesystem object
            output_path: Local output path
            start_byte: Byte offset to start from (for resuming)
            etag: ETag or modification time for content validation
            show_progress: Whether to show progress bar during download
        """
        file_size = fs.size(remote_url)
        mode = "ab" if start_byte > 0 else "wb"
        # Remove .partial suffix to get base cache path for metadata
        base_path = (
            output_path[: -len(".partial")]
            if output_path.endswith(".partial")
            else output_path
        )
        metadata_path = self._get_metadata_path(base_path)

        # Get filename for progress bar description
        filename = os.path.basename(remote_url)

        logger.debug(
            f"Downloading {remote_url} to {output_path} (starting at byte {start_byte})"
        )

        # Create progress bar for download (or dummy context if disabled)
        pbar = (
            tqdm(
                total=file_size,
                initial=start_byte,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {filename}",
                disable=not show_progress,
            )
            if show_progress
            else None
        )

        try:
            with fs.open(remote_url, "rb") as remote:
                if start_byte > 0:
                    remote.seek(start_byte)

                with open(output_path, mode) as local:
                    bytes_written = start_byte
                    chunk_size = 1024 * 1024  # 1MB chunks

                    while True:
                        chunk = remote.read(chunk_size)
                        if not chunk:
                            break
                        local.write(chunk)
                        bytes_written += len(chunk)
                        if pbar:
                            pbar.update(len(chunk))

                        # Update metadata every 10MB for crash recovery
                        if bytes_written % (10 * 1024 * 1024) < chunk_size:
                            self._write_metadata(
                                metadata_path,
                                remote_url,
                                file_size,
                                bytes_written,
                                etag,
                            )
        finally:
            if pbar:
                pbar.close()

    def get_cached_path(
        self, remote_url: str, fs: CloudOrLocalFS, show_progress: bool = True
    ) -> str:
        """
        Get local path to cached file, downloading/resuming if needed.

        This is the main entry point for cache operations. It handles:
        - Returning existing valid cache
        - Resuming interrupted downloads
        - Starting new downloads

        Args:
            remote_url: Remote file URL
            fs: Filesystem object
            show_progress: Whether to show download progress bar

        Returns:
            Local path to cached file
        """
        cache_path = self._get_cache_path(remote_url)
        partial_path = self._get_partial_path(cache_path)
        metadata_path = self._get_metadata_path(cache_path)

        # Already cached and valid?
        if os.path.exists(cache_path):
            if self._validate_cache(cache_path, remote_url, fs):
                logger.debug(f"Using cached file: {cache_path}")
                return cache_path
            else:
                logger.debug(f"Cache invalid, re-downloading: {cache_path}")
                os.remove(cache_path)

        # Resume partial download?
        start_byte = 0
        if os.path.exists(partial_path) and os.path.exists(metadata_path):
            if self._validate_partial(metadata_path, remote_url, fs):
                start_byte = os.path.getsize(partial_path)
                logger.info(f"Resuming download from byte {start_byte}")
            else:
                logger.debug("Cannot resume partial download, starting over")
                os.remove(partial_path)
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                start_byte = 0

        # Download file
        expected_size = fs.size(remote_url)
        etag = self._get_file_etag(remote_url, fs)
        self._write_metadata(metadata_path, remote_url, expected_size, start_byte, etag)
        self._download_file(
            remote_url, fs, partial_path, start_byte, etag, show_progress
        )

        # Finalize: rename partial to final and clean up metadata
        os.rename(partial_path, cache_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)

        logger.info(f"Download complete: {cache_path}")
        return cache_path

    def is_cached(self, remote_url: str, fs: CloudOrLocalFS) -> bool:
        """
        Check if a file is fully cached and valid.

        Args:
            remote_url: Remote file URL
            fs: Filesystem object

        Returns:
            True if file is cached and valid, False otherwise
        """
        cache_path = self._get_cache_path(remote_url)
        return os.path.exists(cache_path) and self._validate_cache(
            cache_path, remote_url, fs
        )

    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache files matching pattern.

        Args:
            pattern: Glob pattern to match files (e.g., "*.parquet"). If None, clears all cache.

        Returns:
            Number of files removed
        """
        count = 0
        cache_path_obj = Path(self.cache_dir)

        if pattern:
            files = cache_path_obj.glob(pattern)
        else:
            files = cache_path_obj.glob("*")

        for file_path in files:
            if file_path.is_file():
                file_path.unlink()
                count += 1
                # Also remove associated metadata/partial files if they exist
                meta_path = Path(str(file_path) + ".meta")
                partial_path = Path(str(file_path) + ".partial")
                if meta_path.exists():
                    meta_path.unlink()
                    count += 1
                if partial_path.exists():
                    partial_path.unlink()
                    count += 1

        logger.info(f"Cleared {count} cache files")
        return count

    def get_cache_info(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache size (bytes), file count, and directory path
        """
        cache_path_obj = Path(self.cache_dir)
        total_size = 0
        file_count = 0

        if cache_path_obj.exists():
            for file_path in cache_path_obj.rglob("*"):
                if file_path.is_file() and not file_path.name.endswith(
                    (".meta", ".partial")
                ):
                    total_size += file_path.stat().st_size
                    file_count += 1

        return {
            "cache_dir": self.cache_dir,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "total_size_gb": round(total_size / (1024 * 1024 * 1024), 2),
            "file_count": file_count,
        }


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get or create the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def set_cache_dir(cache_dir: str) -> None:
    """
    Set the cache directory for the global cache manager.

    Args:
        cache_dir: Directory to store cached files
    """
    global _cache_manager
    _cache_manager = CacheManager(cache_dir=cache_dir)


def cache_info() -> dict:
    """Get cache statistics from the global cache manager."""
    return get_cache_manager().get_cache_info()


def clear_cache(pattern: Optional[str] = None) -> int:
    """
    Clear cache files matching pattern.

    Args:
        pattern: Glob pattern to match files (e.g., "*.parquet"). If None, clears all cache.

    Returns:
        Number of files removed
    """
    return get_cache_manager().clear_cache(pattern)
