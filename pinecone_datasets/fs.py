from typing import Union

import gcsfs
import s3fs
from fsspec.implementations.local import LocalFileSystem

from pinecone_datasets import cfg


def get_cloud_fs(
    path, **kwargs
) -> Union[gcsfs.GCSFileSystem, s3fs.S3FileSystem, LocalFileSystem]:
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
        fs = gcsfs.GCSFileSystem(token="anon" if is_anon else None, **kwargs)
    elif path.startswith("s3://") or "s3.amazonaws.com" in path:
        fs = s3fs.S3FileSystem(anon=is_anon, **kwargs)
    else:
        fs = LocalFileSystem()
    return fs
