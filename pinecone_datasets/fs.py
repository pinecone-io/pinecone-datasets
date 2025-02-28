from typing import Union, TYPE_CHECKING
from importlib import import_module

from pinecone_datasets import cfg

if TYPE_CHECKING:
    import gcsfs
    import s3fs
    from fsspec.implementations.local import LocalFileSystem

    CloudOrLocalFS = Union[gcsfs.GCSFileSystem, s3fs.S3FileSystem, LocalFileSystem]
else:
    CloudOrLocalFS = Union[object]  # type: ignore


def get_cloud_fs(path: str, **kwargs) -> CloudOrLocalFS:
    """
    returns a filesystem object for the given path, if it is a cloud storage path (gs:// or s3://)

    Args:
        path (str): the path to the file or directory
        **kwargs: additional arguments to pass to the filesystem constructor

    Returns:
        fs: Union[gcsfs.GCSFileSystem, s3fs.S3FileSystem] - the filesystem object
    """
    is_anon = path == cfg.Storage.endpoint

    if path.startswith("gs://") or "storage.googleapis.com" in path:
        gcsfs = import_module("gcsfs")
        if kwargs.get("token", None):
            fs = gcsfs.GCSFileSystem(**kwargs)
        else:
            fs = gcsfs.GCSFileSystem(token="anon" if is_anon else None, **kwargs)
    elif path.startswith("s3://") or "s3.amazonaws.com" in path:
        s3fs = import_module("s3fs")
        fs = s3fs.S3FileSystem(anon=is_anon, **kwargs)
    else:
        local_fs = import_module("fsspec.implementations.local")
        fs = local_fs.LocalFileSystem()
    return fs
