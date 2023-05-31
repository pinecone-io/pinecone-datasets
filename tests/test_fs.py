import os
import s3fs
import gcsfs

from pinecone_datasets.fs import get_cloud_fs


def test_get_cloud_fs_nullability():
    assert get_cloud_fs("s3://pinecone-datasets") is not None
    assert get_cloud_fs("gs://pinecone-datasets") is not None
    assert get_cloud_fs("pinecone-datasets") is not None


def test_get_cloud_fs_s3():
    fs = get_cloud_fs("s3://not-pinecone-datasets")
    assert isinstance(fs, s3fs.S3FileSystem)
    assert fs.anon is False


def test_get_cloud_fs_gs():
    fs = get_cloud_fs("gs://not-pinecone-datasets")
    assert isinstance(fs, gcsfs.GCSFileSystem)
    assert fs.credentials.token is None


def test_get_cloud_fs_on_pinecone_endpoint():
    fs = get_cloud_fs("gs://pinecone-datasets-dev")
    assert isinstance(fs, gcsfs.GCSFileSystem)
    assert fs.credentials.token == "anon"
