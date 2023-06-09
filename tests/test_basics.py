import sys
from pinecone_datasets import __version__

if sys.version_info > (3, 11):
    import tomllib as toml
else:
    import toml


def test_version():
    assert __version__ == "0.5.0-rc.1"
    assert toml.load("pyproject.toml")["tool"]["poetry"]["version"] == __version__
