import os
from unittest.mock import patch

import pytest

from pinecone_datasets.retry import (
    RetryConfig,
    create_cloud_storage_retry_decorator,
    is_retryable_error,
)


class TestRetryConfig:
    def test_retry_enabled_by_default(self):
        assert RetryConfig.is_enabled() is True

    def test_retry_disabled_via_env(self):
        with patch.dict(os.environ, {"PINECONE_DATASETS_DISABLE_RETRY": "true"}):
            assert RetryConfig.is_enabled() is False

        with patch.dict(os.environ, {"PINECONE_DATASETS_DISABLE_RETRY": "1"}):
            assert RetryConfig.is_enabled() is False

        with patch.dict(os.environ, {"PINECONE_DATASETS_DISABLE_RETRY": "yes"}):
            assert RetryConfig.is_enabled() is False

    def test_default_max_attempts(self):
        assert RetryConfig.get_max_attempts() == 3

    def test_custom_max_attempts(self):
        with patch.dict(os.environ, {"PINECONE_DATASETS_MAX_RETRY_ATTEMPTS": "5"}):
            assert RetryConfig.get_max_attempts() == 5

    def test_default_wait_times(self):
        assert RetryConfig.get_min_wait() == 2
        assert RetryConfig.get_max_wait() == 10

    def test_custom_wait_times(self):
        with patch.dict(
            os.environ,
            {
                "PINECONE_DATASETS_MIN_RETRY_WAIT": "1",
                "PINECONE_DATASETS_MAX_RETRY_WAIT": "20",
            },
        ):
            assert RetryConfig.get_min_wait() == 1
            assert RetryConfig.get_max_wait() == 20


class TestRetryableErrors:
    def test_connection_errors_are_retryable(self):
        assert is_retryable_error(ConnectionError("Connection failed")) is True
        assert is_retryable_error(TimeoutError("Timeout")) is True
        assert is_retryable_error(ConnectionResetError("Reset")) is True
        assert is_retryable_error(ConnectionAbortedError("Aborted")) is True
        assert is_retryable_error(ConnectionRefusedError("Refused")) is True
        assert is_retryable_error(BrokenPipeError("Broken pipe")) is True

    def test_non_retryable_errors(self):
        assert is_retryable_error(FileNotFoundError("Not found")) is False
        assert is_retryable_error(PermissionError("Permission denied")) is False
        assert is_retryable_error(ValueError("Invalid value")) is False
        assert is_retryable_error(KeyError("Key not found")) is False

    def test_transient_message_errors_are_retryable(self):
        assert is_retryable_error(Exception("Network timeout")) is True
        assert is_retryable_error(Exception("Connection reset by peer")) is True
        assert is_retryable_error(Exception("Service unavailable")) is True
        assert is_retryable_error(Exception("Rate limit exceeded")) is True
        assert is_retryable_error(Exception("Too many requests")) is True


class TestRetryDecorator:
    def test_decorator_retries_on_transient_error(self):
        retry_decorator = create_cloud_storage_retry_decorator()
        call_count = 0

        @retry_decorator
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary network issue")
            return "success"

        result = failing_function()
        assert result == "success"
        assert call_count == 3

    def test_decorator_does_not_retry_on_permanent_error(self):
        retry_decorator = create_cloud_storage_retry_decorator()
        call_count = 0

        @retry_decorator
        def failing_function():
            nonlocal call_count
            call_count += 1
            raise FileNotFoundError("File does not exist")

        with pytest.raises(FileNotFoundError):
            failing_function()

        assert call_count == 1

    def test_decorator_disabled_via_env(self):
        with patch.dict(os.environ, {"PINECONE_DATASETS_DISABLE_RETRY": "true"}):
            retry_decorator = create_cloud_storage_retry_decorator()
            call_count = 0

            @retry_decorator
            def failing_function():
                nonlocal call_count
                call_count += 1
                raise ConnectionError("Network issue")

            with pytest.raises(ConnectionError):
                failing_function()

            assert call_count == 1

    def test_decorator_stops_after_max_attempts(self):
        with patch.dict(os.environ, {"PINECONE_DATASETS_MAX_RETRY_ATTEMPTS": "2"}):
            retry_decorator = create_cloud_storage_retry_decorator()
            call_count = 0

            @retry_decorator
            def failing_function():
                nonlocal call_count
                call_count += 1
                raise ConnectionError("Persistent network issue")

            with pytest.raises(ConnectionError):
                failing_function()

            assert call_count == 2
