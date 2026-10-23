import tomllib

from pinecone_datasets import __version__

with open("pyproject.toml", "rb") as f:
    assert tomllib.load(f)["project"]["version"] == __version__


def test_version():
    assert __version__ == "1.1.0"
