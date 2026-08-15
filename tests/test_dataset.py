import pandas as pd
import pytest

from constants import LOCAL, PROCESSED_DATA_PATH
from src.dataset import Dataset


def test_transform_keeps_only_doc_id_title_text():
    ds = Dataset(LOCAL, dataset_size=5)
    row = {"id": "abc123", "title": "Some Title", "text": "Some text.", "extra": "drop me"}

    out = ds.transform(row)

    assert out == {"doc_id": "abc123", "title": "Some Title", "text": "Some text."}


def test_load_parquet_dataset_local_reads_existing_file(isolated_cwd):
    dataset_size = 3
    path = isolated_cwd / f"{PROCESSED_DATA_PATH}/{dataset_size}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"doc_id": "d1", "title": "T1", "text": "hello world"},
            {"doc_id": "d2", "title": "T2", "text": "goodbye world"},
        ]
    ).to_parquet(path)

    ds = Dataset(LOCAL, dataset_size)
    loaded = ds.load_parquet_dataset_local()

    assert loaded is not None
    rows = list(loaded)
    assert len(rows) == 2
    assert {r["doc_id"] for r in rows} == {"d1", "d2"}


def test_load_parquet_dataset_local_returns_none_on_creation_failure(isolated_cwd, monkeypatch):
    ds = Dataset(LOCAL, dataset_size=3)
    assert not (isolated_cwd / ds.path).exists()

    def boom(*args, **kwargs):
        raise RuntimeError("no source data available")

    monkeypatch.setattr("src.dataset.load_dataset", boom)

    assert ds.load_parquet_dataset_local() is None


def test_load_parquet_dataset_local_returns_none_when_write_produces_no_file(isolated_cwd, monkeypatch):
    ds = Dataset(LOCAL, dataset_size=3)

    class FakeHFDataset:
        column_names = ["id", "title", "text"]

        def filter(self, fn):
            return self

        def shuffle(self, seed):
            return self

        def select(self, rng):
            return self

        def map(self, fn, remove_columns=None):
            return self

        def to_parquet(self, path):
            pass  # deliberately does not create a file, mirroring a silent write failure

    monkeypatch.setattr("src.dataset.load_dataset", lambda *a, **k: FakeHFDataset())

    assert ds.load_parquet_dataset_local() is None
