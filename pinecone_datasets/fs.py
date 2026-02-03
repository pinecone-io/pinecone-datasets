from importlib import import_module
from typing import TYPE_CHECKING, Optional, Union

from pinecone_datasets import cfg

if TYPE_CHECKING:
    import gcsfs
    import s3fs
    from fsspec.implementations.local import LocalFileSystem

    CloudOrLocalFS = Union[gcsfs.GCSFileSystem, s3fs.S3FileSystem, LocalFileSystem]
else:
    CloudOrLocalFS = Union[object]  # type: ignore


def is_cloud_path(path: str) -> bool:
    """
    Check if a path is a cloud storage path.

    Args:
        path: File or directory path

    Returns:
        True if path is cloud storage (GCS or S3), False otherwise
    """
    return (
        path.startswith("gs://")
        or path.startswith("s3://")
        or path.startswith("https://storage.googleapis.com/")
        or path.startswith("https://s3.amazonaws.com/")
    )


def should_use_cache(path: str, use_cache: Optional[bool] = None) -> bool:
    """
    Determine if caching should be used for a given path.

    Args:
        path: File or directory path
        use_cache: Explicit cache preference. If None, defaults based on path type.

    Returns:
        True if caching should be used, False otherwise
    """
    if use_cache is not None:
        return use_cache

    # Default: use cache for cloud paths if enabled in config
    if is_cloud_path(path):
        return cfg.Cache.use_cache

    # Don't cache local paths by default
    return False


def get_cloud_fs(path: str, **kwargs) -> CloudOrLocalFS:
    """
    returns a filesystem object for the given path, if it is a cloud storage path (gs:// or s3://)

    Args:
        path (str): the path to the file or directory
        **kwargs: additional arguments to pass to the filesystem constructor

    Returns:
        fs: Union[gcsfs.GCSFileSystem, s3fs.S3FileSystem, LocalFileSystem] - the filesystem object
    """
    is_anon = path == cfg.Storage.endpoint

    if path.startswith("gs://") or path.startswith("https://storage.googleapis.com/"):
        gcsfs = import_module("gcsfs")
        if kwargs.get("token", None):
            fs = gcsfs.GCSFileSystem(**kwargs)
        else:
            fs = gcsfs.GCSFileSystem(token="anon" if is_anon else None, **kwargs)
    elif path.startswith("s3://") or path.startswith("https://s3.amazonaws.com/"):
        s3fs = import_module("s3fs")
        fs = s3fs.S3FileSystem(anon=is_anon, **kwargs)
    else:
        local_fs = import_module("fsspec.implementations.local")
        fs = local_fs.LocalFileSystem()
    return fs


def get_cached_path(
    path: str,
    fs: CloudOrLocalFS,
    use_cache: Optional[bool] = None,
    show_progress: bool = True,
) -> str:
    """
    Get local path to file, using cache if appropriate.

    For cloud paths with caching enabled, this downloads the file to local cache
    (with resume support) and returns the local path. For local paths or when
    caching is disabled, returns the original path.

    Args:
        path: Remote or local file path
        fs: Filesystem object
        use_cache: Whether to use caching. If None, uses default based on path type.
        show_progress: Whether to show download progress bar. Useful to disable
            during parallel downloads to avoid visual clutter.

    Returns:
        Local file path (either cached or original)
    """
    if should_use_cache(path, use_cache):
        from .cache import get_cache_manager

        cache_manager = get_cache_manager()
        return cache_manager.get_cached_path(path, fs, show_progress=show_progress)
    return path
