import pytest

from pinecone_datasets.dataset_metadata import DatasetMetadata, DenseModelMetadata

from pydantic import ValidationError


def test_metadata_fields_minimal():
    try:
        meta = DatasetMetadata(
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
        meta = DatasetMetadata(
            documents=1,
            queries=1,
            dense_model=DenseModelMetadata(
                name="ada2",
                dimensions=2,
            ),
        )


def test_validation_error_optional_field():
    with pytest.raises(ValidationError):
        meta = DatasetMetadata(
            name="test",
            documents=1,
            queries=1,
            dense_model=DenseModelMetadata(name="ada2", dimension=2),
            tags="test",
        )
