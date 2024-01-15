import pinecone
from packaging import version


def is_pinecone_3() -> bool:
    return version.parse(pinecone.__version__) >= version.parse("3.0.0.dev0")
