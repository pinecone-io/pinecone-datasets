import sys

from pinecone_datasets import __version__

if sys.version_info > (3, 11):
    import tomllib as toml

    with open("pyproject.toml", "rb") as f:
        assert toml.load(f)["project"]["version"] == __version__
else:
    import toml

    with open("pyproject.toml") as f:
        assert toml.load(f)["project"]["version"] == __version__


def test_version():
    assert __version__ == "1.0.2"
