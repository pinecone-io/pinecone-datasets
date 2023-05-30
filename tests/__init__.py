import os
import pytest

if "S3_ACCESS_KEY" not in os.environ or "S3_SECRET" not in os.environ:
    pytest.fail("S3_ACCESS_KEY and S3_SECRET must be set as environment variables")
