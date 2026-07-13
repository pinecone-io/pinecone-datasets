from datetime import datetime
from typing import Any

from pydantic import BaseModel


class DenseModelMetadata(BaseModel):
    name: str
    tokenizer: str | None = None
    dimension: int


class SparseModelMetdata(BaseModel):
    name: str | None = None
    tokenizer: str | None = None


def get_time_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


class DatasetMetadata(BaseModel):
    name: str
    created_at: str
    documents: int
    queries: int
    source: str | None = None
    license: str | None = None
    bucket: str | None = None
    task: str | None = None
    dense_model: DenseModelMetadata
    sparse_model: SparseModelMetdata | None = None
    description: str | None = None
    tags: list[str] | None = None
    args: dict[str, Any] | None = None

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
