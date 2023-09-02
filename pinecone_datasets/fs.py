from typing import Union, Optional

import gcsfs
import s3fs
from fsspec.implementations.local import LocalFileSystem

from pinecone_datasets import cfg


def get_cloud_fs(
    endpoint: Optional[str] = None, **kwargs
) -> Union[gcsfs.GCSFileSystem, s3fs.S3FileSystem, LocalFileSystem]:
    """
    returns a filesystem object for the given path, if it is a cloud 
    storage path (gs:// or s3:// or custom s3 compatible http endpoint
    such as ClodFlare R2 or minio)

    Parameters
    ----------
    endpoint : string
        Input path, like:
        - `s3://mybucket`    
        - `https://{ACCOUNT_ID}.r2.cloudflarestorage.com/{BUCKET_NAME}`
        - `gs://mybucket`
    
    **kwargs: 
        Additional arguments to pass to the filesystem constructor,
        can be either:
        - `gcsfs.GCSFileSystem`
        - `s3fs.S3FileSystem`
        - `LocalFileSystem`

    Returns
    -------
    Union[gcsfs.GCSFileSystem, s3fs.S3FileSystem, LocalFileSystem]
    """
    if endpoint:
        is_anon = endpoint == cfg.Storage.endpoint

        if endpoint.startswith("gs://") or "storage.googleapis.com" in endpoint:
            return gcsfs.GCSFileSystem(token="anon" if is_anon else None, **kwargs)
        elif endpoint.startswith("s3://") or "s3.amazonaws.com" in endpoint:
            return s3fs.S3FileSystem(anon=is_anon, **kwargs)
        elif endpoint.startswith("http"):
            return s3fs.S3FileSystem(**kwargs)
        
    return LocalFileSystem(**kwargs)
