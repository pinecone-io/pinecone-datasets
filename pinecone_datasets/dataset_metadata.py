from datetime import datetime
from typing import List, Optional, Any, Dict
from pydantic import BaseModel


class DenseModelMetadata(BaseModel):
    name: str
    tokenizer: Optional[str] = None
    dimension: int


class SparseModelMetdata(BaseModel):
    name: Optional[str] = None
    tokenizer: Optional[str] = None


def get_time_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


class DatasetMetadata(BaseModel):
    name: str
    created_at: str
    documents: int
    queries: int
    source: Optional[str] = None
    license: Optional[str] = None
    bucket: Optional[str] = None
    task: Optional[str] = None
    dense_model: DenseModelMetadata
    sparse_model: Optional[SparseModelMetdata] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    args: Optional[Dict[str, Any]] = None

    @staticmethod
    def empty() -> "DatasetMetadata":
        return DatasetMetadata(
            name="",
            created_at=get_time_now(),
            documents=0,
            queries=0,
            dense_model=DenseModelMetadata(name="", dimension=0),
        )

    def is_empty(self) -> bool:
        return self.name == "" and self.documents == 0 and self.queries == 0
