import pytest
from pydantic import ValidationError

from pinecone_datasets.dataset_metadata import DatasetMetadata, DenseModelMetadata


def test_metadata_fields_minimal():
    try:
        DatasetMetadata(
            name="test",
            documents=1,
            created_at="2021-01-01 00:00:00.000000",
            queries=1,
            dense_model=DenseModelMetadata(
                name="ada2",
                dimension=2,
            ),
        )
    except NameError:
        pytest.fail("Validation error")


def test_validation_error_mandatory_field():
    with pytest.raises(ValidationError):
        DatasetMetadata(
            documents=1,
            queries=1,
            dense_model=DenseModelMetadata(
                name="ada2",
                dimensions=2,
            ),
        )


def test_validation_error_optional_field():
    with pytest.raises(ValidationError):
        DatasetMetadata(
            name="test",
            documents=1,
            queries=1,
            dense_model=DenseModelMetadata(name="ada2", dimension=2),
            tags="test",
        )
