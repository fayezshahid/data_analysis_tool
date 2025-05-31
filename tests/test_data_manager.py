import os
import tempfile
import pandas as pd
import pytest
from core.data_manager import DatasetManager

# Dummy constant for testing
from core.constants import SUPPORTED_FORMATS

@pytest.fixture
def temp_dirs():
    """Create temporary dataset and metadata directories."""
    with tempfile.TemporaryDirectory() as dataset_dir, tempfile.TemporaryDirectory() as metadata_dir:
        yield dataset_dir, metadata_dir


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing."""
    sample_file = tmp_path / "sample.csv"
    df = pd.DataFrame({
        "name": ["Alice", "Bob"],
        "age": [30, 25]
    })
    df.to_csv(sample_file, index=False)
    return str(sample_file)

def test_load_dataset_success(temp_dirs, sample_csv):
    dataset_dir, metadata_dir = temp_dirs
    manager = DatasetManager(dataset_dir=dataset_dir, metadata_dir=metadata_dir)

    result = manager.load_dataset(sample_csv, "test_dataset")
    assert result is True
    assert "test_dataset" in manager.datasets
    assert "test_dataset" in manager.metadata


def test_load_dataset_empty_name(temp_dirs, sample_csv):
    dataset_dir, metadata_dir = temp_dirs
    manager = DatasetManager(dataset_dir=dataset_dir, metadata_dir=metadata_dir)

    result = manager.load_dataset(sample_csv, "")
    assert result is False


def test_load_dataset_duplicate_name(temp_dirs, sample_csv):
    dataset_dir, metadata_dir = temp_dirs
    manager = DatasetManager(dataset_dir=dataset_dir, metadata_dir=metadata_dir)

    manager.load_dataset(sample_csv, "duplicate")
    result = manager.load_dataset(sample_csv, "duplicate")
    assert result is False


def test_load_dataset_invalid_file(temp_dirs):
    dataset_dir, metadata_dir = temp_dirs
    manager = DatasetManager(dataset_dir=dataset_dir, metadata_dir=metadata_dir)

    result = manager.load_dataset("nonexistent.csv", "badfile")
    assert result is False


def test_remove_dataset_success(temp_dirs, sample_csv):
    dataset_dir, metadata_dir = temp_dirs
    manager = DatasetManager(dataset_dir=dataset_dir, metadata_dir=metadata_dir)

    manager.load_dataset(sample_csv, "toremove")
    result = manager.remove_dataset("toremove")
    assert result is True
    assert "toremove" not in manager.metadata
    assert "toremove" not in manager.datasets


def test_remove_dataset_not_found(temp_dirs):
    dataset_dir, metadata_dir = temp_dirs
    manager = DatasetManager(dataset_dir=dataset_dir, metadata_dir=metadata_dir)

    result = manager.remove_dataset("ghost")
    assert result is False
