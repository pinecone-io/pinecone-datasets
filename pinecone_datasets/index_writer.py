import time
from pinecone_datasets.utils import deprecated
from pinecone_datasets.cfg import Schema
from pinecone import Pinecone, ServerlessSpec, PodSpec
from typing import Optional, List, NamedTuple
import pandas as pd


class IndexWriter:
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self._pc = Pinecone(api_key=api_key, **kwargs)

    def create_serverless_index(
        self,
        index_name: str,
        dimension: int,
        region: Optional[str] = None,
        cloud: Optional[str] = None,
        **kwargs,
    ):
        if self._pc.has_index(index_name):
            raise ValueError(
                f"index {index_name} already exists, Pinecone Datasets can only be upserted to a new index"
            )
        self._pc.create_index(
            name=index_name,
            dimension=dimension,
            spec=ServerlessSpec(
                cloud=cloud,
                region=region,
            ),
            **kwargs,
        )
        self._wait_for_index_creation(index_name=index_name)

    def create_pod_index(
        self,
        index_name: str,
        dimension: int,
        environment: Optional[str] = None,
        **kwargs,
    ):
        if self._pc.has_index(index_name):
            raise ValueError(
                f"index {index_name} already exists, Pinecone Datasets can only be upserted to a new indexe"
            )
        self._pc.create_index(
            name=index_name,
            dimension=dimension,
            spec=PodSpec(environment=environment),
            **kwargs,
        )
        self._wait_for_index_creation(index_name=index_name)

    def upsert_to_index(
        self,
        index_name: str,
        df: pd.DataFrame,
        namespace: str,
        batch_size: int,
        show_progress: bool,
        **kwargs,
    ):
        pinecone_index = self._pc.Index(name=index_name)
        res = pinecone_index.upsert_from_dataframe(
            df[Schema.documents_select_columns].dropna(axis=1, how="all"),
            namespace=namespace,
            batch_size=batch_size,
            show_progress=show_progress,
            **kwargs,
        )
        return {"upserted_count": res.upserted_count}

    def _wait_for_index_creation(self, index_name: str, timeout: int = 60):
        for _ in range(timeout):
            try:
                self._pc.Index(name=index_name).describe_index_stats()
                return
            except Exception as e:
                time.sleep(1)
        raise TimeoutError(f"Index creation timed out after {timeout} seconds")
