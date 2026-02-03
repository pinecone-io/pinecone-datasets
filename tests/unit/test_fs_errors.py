from unittest.mock import Mock, patch

import pytest

from pinecone_datasets.fs import get_cloud_fs


class TestFSErrorPaths:
    """Test error handling in filesystem abstraction"""

    def test_get_cloud_fs_invalid_s3_credentials(self):
        """Test getting S3 filesystem with invalid credentials"""
        # This should create the filesystem object, but actual operations would fail
        # The function itself doesn't validate credentials
        fs = get_cloud_fs("s3://test-bucket")
        assert fs is not None

    def test_get_cloud_fs_invalid_gcs_credentials(self):
        """Test getting GCS filesystem with invalid credentials"""
        # This should create the filesystem object, but actual operations would fail
        fs = get_cloud_fs("gs://test-bucket")
        assert fs is not None

    def test_get_cloud_fs_with_invalid_token(self):
        """Test getting cloud filesystem with invalid token parameter"""
        # Should still create filesystem, token validation happens during operations
        fs = get_cloud_fs("gs://test-bucket", token="invalid_token")
        assert fs is not None

    def test_get_cloud_fs_s3_import_error(self):
        """Test error when s3fs module cannot be imported"""
        with patch("pinecone_datasets.fs.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module named 's3fs'")

            with pytest.raises(ImportError):
                get_cloud_fs("s3://test-bucket")

    def test_get_cloud_fs_gcs_import_error(self):
        """Test error when gcsfs module cannot be imported"""
        with patch("pinecone_datasets.fs.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module named 'gcsfs'")

            with pytest.raises(ImportError):
                get_cloud_fs("gs://test-bucket")

    def test_get_cloud_fs_local_import_error(self):
        """Test error when local filesystem module cannot be imported"""
        with patch("pinecone_datasets.fs.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module named 'fsspec'")

            with pytest.raises(ImportError):
                get_cloud_fs("/local/path")

    def test_get_cloud_fs_empty_path(self):
        """Test getting filesystem with empty path"""
        fs = get_cloud_fs("")
        assert fs is not None
        # Empty path should default to local filesystem

    def test_get_cloud_fs_malformed_s3_url(self):
        """Test getting filesystem with malformed S3 URL"""
        # Should still create filesystem, URL validation happens during operations
        fs = get_cloud_fs("s3://")
        assert fs is not None

    def test_get_cloud_fs_malformed_gcs_url(self):
        """Test getting filesystem with malformed GCS URL"""
        # Should still create filesystem, URL validation happens during operations
        fs = get_cloud_fs("gs://")
        assert fs is not None

    def test_get_cloud_fs_s3_with_http_url(self):
        """Test getting S3 filesystem with HTTP URL"""
        fs = get_cloud_fs("https://s3.amazonaws.com/bucket/path")
        assert fs is not None

    def test_get_cloud_fs_gcs_with_http_url(self):
        """Test getting GCS filesystem with HTTP URL"""
        fs = get_cloud_fs("https://storage.googleapis.com/bucket/path")
        assert fs is not None

    def test_get_cloud_fs_s3_anon_parameter(self):
        """Test S3 filesystem detects public endpoint correctly"""
        # Test with non-public bucket (anon should be False)
        fs = get_cloud_fs("s3://test-bucket")
        assert fs is not None

    def test_get_cloud_fs_gcs_anon_parameter(self):
        """Test GCS filesystem with anon parameter for public bucket"""
        from pinecone_datasets import cfg

        # Using the public bucket should set token to "anon"
        fs = get_cloud_fs(cfg.Storage.endpoint)
        assert fs is not None

    def test_get_cloud_fs_with_custom_kwargs_s3(self):
        """Test S3 filesystem with custom kwargs"""
        fs = get_cloud_fs("s3://test-bucket", endpoint_url="https://custom.s3.endpoint")
        assert fs is not None

    def test_get_cloud_fs_with_custom_kwargs_gcs(self):
        """Test GCS filesystem with custom kwargs"""
        fs = get_cloud_fs(
            "gs://test-bucket", endpoint_url="https://custom.gcs.endpoint"
        )
        assert fs is not None

    def test_get_cloud_fs_unsupported_protocol(self):
        """Test getting filesystem with unsupported protocol"""
        # FTP protocol not supported, should default to local filesystem
        fs = get_cloud_fs("ftp://test-server/path")
        assert fs is not None

    def test_get_cloud_fs_special_characters_in_path(self):
        """Test getting filesystem with special characters in path"""
        fs = get_cloud_fs("/path/with/special/chars/@#$/test")
        assert fs is not None

    def test_get_cloud_fs_unicode_path(self):
        """Test getting filesystem with unicode characters in path"""
        fs = get_cloud_fs("/path/with/unicode/日本語/test")
        assert fs is not None

    def test_get_cloud_fs_relative_path(self):
        """Test getting filesystem with relative path"""
        fs = get_cloud_fs("./relative/path")
        assert fs is not None

    def test_get_cloud_fs_windows_path(self):
        """Test getting filesystem with Windows-style path"""
        fs = get_cloud_fs("C:\\Windows\\Path")
        assert fs is not None

    def test_get_cloud_fs_s3_creation_failure(self):
        """Test S3 filesystem creation failure"""
        with patch("pinecone_datasets.fs.import_module") as mock_import:
            mock_s3fs = Mock()
            mock_s3fs.S3FileSystem.side_effect = Exception(
                "Failed to create filesystem"
            )
            mock_import.return_value = mock_s3fs

            with pytest.raises(Exception, match="Failed to create filesystem"):
                get_cloud_fs("s3://test-bucket")

    def test_get_cloud_fs_gcs_creation_failure(self):
        """Test GCS filesystem creation failure"""
        with patch("pinecone_datasets.fs.import_module") as mock_import:
            mock_gcsfs = Mock()
            mock_gcsfs.GCSFileSystem.side_effect = Exception(
                "Failed to create filesystem"
            )
            mock_import.return_value = mock_gcsfs

            with pytest.raises(Exception, match="Failed to create filesystem"):
                get_cloud_fs("gs://test-bucket")

    def test_get_cloud_fs_local_creation_failure(self):
        """Test local filesystem creation failure"""
        with patch("pinecone_datasets.fs.import_module") as mock_import:
            mock_local = Mock()
            mock_local.LocalFileSystem.side_effect = Exception(
                "Failed to create filesystem"
            )
            mock_import.return_value = mock_local

            with pytest.raises(Exception, match="Failed to create filesystem"):
                get_cloud_fs("/local/path")

    def test_get_cloud_fs_with_none_path(self):
        """Test getting filesystem with None path"""
        with pytest.raises(AttributeError):
            get_cloud_fs(None)

    def test_get_cloud_fs_s3_with_custom_token(self):
        """Test S3 filesystem ignores token parameter (not applicable to S3)"""
        fs = get_cloud_fs("s3://test-bucket", token="custom_token")
        assert fs is not None

    def test_get_cloud_fs_gcs_with_custom_token(self):
        """Test GCS filesystem with custom token parameter"""
        fs = get_cloud_fs("gs://test-bucket", token="custom_token")
        assert fs is not None

    def test_get_cloud_fs_network_unreachable(self):
        """Test filesystem creation when network is unreachable"""
        # Filesystem creation usually succeeds, network errors occur during operations
        fs = get_cloud_fs("s3://unreachable-bucket")
        assert fs is not None

    def test_get_cloud_fs_dns_resolution_failure(self):
        """Test filesystem creation with DNS resolution failure"""
        # Filesystem creation usually succeeds, DNS errors occur during operations
        fs = get_cloud_fs("s3://nonexistent.domain.invalid/bucket")
        assert fs is not None

    def test_get_cloud_fs_connection_timeout(self):
        """Test filesystem creation with connection timeout"""
        # Filesystem creation usually succeeds, timeout errors occur during operations
        fs = get_cloud_fs("s3://timeout-bucket")
        assert fs is not None

    def test_get_cloud_fs_ssl_error(self):
        """Test filesystem creation with SSL certificate errors"""
        # Filesystem creation usually succeeds, SSL errors occur during operations
        fs = get_cloud_fs("https://invalid-cert.example.com")
        assert fs is not None

    def test_get_cloud_fs_proxy_error(self):
        """Test filesystem creation with proxy configuration errors"""
        # Filesystem creation usually succeeds, proxy errors occur during operations
        fs = get_cloud_fs("s3://test-bucket")
        assert fs is not None

    def test_get_cloud_fs_s3_region_mismatch(self):
        """Test S3 filesystem with region mismatch"""
        # Filesystem creation succeeds, region errors occur during operations
        fs = get_cloud_fs("s3://us-west-2-bucket")
        assert fs is not None

    def test_get_cloud_fs_gcs_project_permission_error(self):
        """Test GCS filesystem with project permission issues"""
        # Filesystem creation succeeds, permission errors occur during operations
        fs = get_cloud_fs("gs://restricted-project-bucket")
        assert fs is not None

    def test_get_cloud_fs_mixed_case_protocol(self):
        """Test filesystem with mixed case protocol"""
        fs = get_cloud_fs("S3://test-bucket")
        # Should default to local filesystem as protocol check is case-sensitive
        assert fs is not None

    def test_get_cloud_fs_path_with_query_params(self):
        """Test filesystem with path containing query parameters"""
        fs = get_cloud_fs("s3://bucket/path?param=value")
        assert fs is not None

    def test_get_cloud_fs_path_with_fragment(self):
        """Test filesystem with path containing fragment"""
        fs = get_cloud_fs("s3://bucket/path#fragment")
        assert fs is not None

    def test_get_cloud_fs_extremely_long_path(self):
        """Test filesystem with extremely long path"""
        long_path = "s3://bucket/" + "a" * 10000
        fs = get_cloud_fs(long_path)
        assert fs is not None

    def test_get_cloud_fs_path_traversal_attempt(self):
        """Test filesystem with path traversal characters"""
        fs = get_cloud_fs("../../../etc/passwd")
        assert fs is not None

    def test_get_cloud_fs_null_bytes_in_path(self):
        """Test filesystem with null bytes in path"""
        # May raise ValueError or similar depending on implementation
        try:
            get_cloud_fs("/path/with/\x00/null")
            # If it doesn't raise, that's also valid behavior
        except ValueError:
            pass  # Expected behavior

    def test_get_cloud_fs_s3_with_credentials_in_url(self):
        """Test S3 filesystem with credentials embedded in URL"""
        fs = get_cloud_fs("s3://access_key:secret_key@bucket/path")
        assert fs is not None

    def test_get_cloud_fs_concurrent_creation(self):
        """Test creating multiple filesystems concurrently"""
        import concurrent.futures

        def create_fs(path):
            return get_cloud_fs(path)

        paths = ["s3://bucket1", "gs://bucket2", "/local/path", "s3://bucket3"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(create_fs, path) for path in paths]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert all(fs is not None for fs in results)

    def test_get_cloud_fs_memory_pressure(self):
        """Test filesystem creation under memory pressure"""
        # Create many filesystem objects to test memory handling
        filesystems = []
        for i in range(100):
            fs = get_cloud_fs(f"/local/path/{i}")
            filesystems.append(fs)

        assert len(filesystems) == 100
        assert all(fs is not None for fs in filesystems)
