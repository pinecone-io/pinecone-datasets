import os
import pytest

if "AWS_ACCESS_KEY_ID"  not in os.environ or "AWS_SECRET_ACCESS_KEY" not in os.environ:
    pytest.fail("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set as environment variables")
