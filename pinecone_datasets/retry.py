import logging
import os
from typing import Callable

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior in cloud storage operations."""

    @staticmethod
    def is_enabled() -> bool:
        """Check if retry is enabled via environment variable."""
        return os.environ.get(
            "PINECONE_DATASETS_DISABLE_RETRY", "false"
        ).lower() not in (
            "true",
            "1",
            "yes",
        )

    @staticmethod
    def get_max_attempts() -> int:
        """Get maximum retry attempts from environment or use default."""
        try:
            return int(os.environ.get("PINECONE_DATASETS_MAX_RETRY_ATTEMPTS", "3"))
        except ValueError:
            return 3

    @staticmethod
    def get_min_wait() -> int:
        """Get minimum wait time in seconds from environment or use default."""
        try:
            return int(os.environ.get("PINECONE_DATASETS_MIN_RETRY_WAIT", "2"))
        except ValueError:
            return 2

    @staticmethod
    def get_max_wait() -> int:
        """Get maximum wait time in seconds from environment or use default."""
        try:
            return int(os.environ.get("PINECONE_DATASETS_MAX_RETRY_WAIT", "10"))
        except ValueError:
            return 10


def is_retryable_error(exception: Exception) -> bool:
    """
    Determine if an exception is a transient error that should be retried.

    Retryable errors include:
    - Network errors (ConnectionError, TimeoutError)
    - Temporary cloud storage errors
    - OSError with specific errno values

    Non-retryable errors include:
    - FileNotFoundError (404)
    - PermissionError (403)
    - ValueError (invalid input)
    """
    retryable_types = (
        ConnectionError,
        TimeoutError,
        ConnectionResetError,
        ConnectionAbortedError,
        ConnectionRefusedError,
        BrokenPipeError,
    )

    if isinstance(exception, retryable_types):
        return True

    if isinstance(exception, OSError):
        import errno

        retryable_errno = {
            errno.ETIMEDOUT,
            errno.ECONNRESET,
            errno.ECONNREFUSED,
            errno.ECONNABORTED,
            errno.ENETUNREACH,
            errno.EHOSTUNREACH,
        }
        if hasattr(exception, "errno") and exception.errno in retryable_errno:
            return True

    exception_str = str(exception).lower()
    transient_messages = [
        "timeout",
        "timed out",
        "connection reset",
        "connection refused",
        "temporary failure",
        "service unavailable",
        "too many requests",
        "rate limit",
        "network",
        "unreachable",
    ]
    if any(msg in exception_str for msg in transient_messages):
        return True

    return False


def log_retry_attempt(retry_state: RetryCallState) -> None:
    """Log retry attempts for debugging."""
    if retry_state.attempt_number > 1:
        exception = retry_state.outcome.exception()
        logger.warning(
            f"Retry attempt {retry_state.attempt_number} for {retry_state.fn.__name__} "
            f"after error: {type(exception).__name__}: {exception}"
        )


def create_cloud_storage_retry_decorator() -> Callable:
    """
    Create a retry decorator for cloud storage operations.

    Returns a decorator that can be applied to functions performing cloud storage operations.
    The retry behavior is configurable via environment variables.
    """
    config = RetryConfig()

    if not config.is_enabled():

        def no_retry(func):
            return func

        return no_retry

    return retry(
        stop=stop_after_attempt(config.get_max_attempts()),
        wait=wait_exponential(
            multiplier=1,
            min=config.get_min_wait(),
            max=config.get_max_wait(),
        ),
        retry=retry_if_exception(is_retryable_error),
        before_sleep=log_retry_attempt,
        reraise=True,
    )
