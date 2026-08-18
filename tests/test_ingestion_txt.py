import os
import shutil

import pytest

from src.ingestion.ingestion import ingest_directory

PG19_SAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw", "pg19_sample",
)


def pg19_sample_available():
    if not os.path.isdir(PG19_SAMPLE_DIR):
        return False
    txt_filenames = [f for f in os.listdir(PG19_SAMPLE_DIR) if f.endswith(".txt")]
    return len(txt_filenames) >= 25


pytestmark = pytest.mark.skipif(
    not pg19_sample_available(),
    reason="pg19 sample corpus not found; run scripts/download_pg19_sample.py first",
)


@pytest.fixture()
def pg19_corpus_with_fixtures(tmp_path):
    corpus_dir = tmp_path / "pg19_corpus"
    corpus_dir.mkdir()

    source_filenames = sorted(f for f in os.listdir(PG19_SAMPLE_DIR) if f.endswith(".txt"))[:25]
    for filename in source_filenames:
        shutil.copyfile(os.path.join(PG19_SAMPLE_DIR, filename), corpus_dir / filename)

    metadata_sidecar_path = os.path.join(PG19_SAMPLE_DIR, "metadata.json")
    if os.path.isfile(metadata_sidecar_path):
        shutil.copyfile(metadata_sidecar_path, corpus_dir / "metadata.json")

    first_filename = source_filenames[0]
    first_file_text = (corpus_dir / first_filename).read_text(encoding="utf-8")

    exact_duplicate_filename = "exact_duplicate_of_" + first_filename
    (corpus_dir / exact_duplicate_filename).write_text(first_file_text, encoding="utf-8")

    paragraphs = first_file_text.split("\n\n")
    assert len(paragraphs) >= 3, f"expected multiple paragraphs in {first_filename}"
    middle_index = len(paragraphs) // 2
    near_duplicate_paragraphs = paragraphs[:middle_index] + paragraphs[middle_index + 1:]
    near_duplicate_text = "\n\n".join(near_duplicate_paragraphs)
    near_duplicate_filename = "near_duplicate_of_" + first_filename
    (corpus_dir / near_duplicate_filename).write_text(near_duplicate_text, encoding="utf-8")

    second_filename = source_filenames[1]
    second_file_text = (corpus_dir / second_filename).read_text(encoding="utf-8")
    truncated_filename = "truncated_book.txt"
    (corpus_dir / truncated_filename).write_text(
        second_file_text[:len(second_file_text) // 2], encoding="utf-8"
    )

    third_filename = source_filenames[2]
    third_file_bytes = (corpus_dir / third_filename).read_bytes()
    invalid_utf8_filename = "invalid_utf8_book.txt"
    corrupted_bytes = third_file_bytes[:1000] + b"\xff\xfe\xfa" + third_file_bytes[1000:]
    (corpus_dir / invalid_utf8_filename).write_bytes(corrupted_bytes)

    zero_byte_filename = "empty_book.txt"
    (corpus_dir / zero_byte_filename).write_bytes(b"")

    return {
        "corpus_dir": str(corpus_dir),
        "first_filename": first_filename,
        "exact_duplicate_filename": exact_duplicate_filename,
        "near_duplicate_filename": near_duplicate_filename,
        "truncated_filename": truncated_filename,
        "invalid_utf8_filename": invalid_utf8_filename,
        "zero_byte_filename": zero_byte_filename,
    }


def doc_id_for_filename(filename):
    return os.path.splitext(filename)[0]


def test_ingest_directory_processes_all_files_without_raising(pg19_corpus_with_fixtures):
    ingest_directory(pg19_corpus_with_fixtures["corpus_dir"])


def test_corrupt_files_are_isolated_not_dropped_or_crashing(pg19_corpus_with_fixtures):
    fixtures = pg19_corpus_with_fixtures
    kept_docs, metadata_records, duplicate_records, failure_records = ingest_directory(
        fixtures["corpus_dir"]
    )

    total_input_files = len(
        [f for f in os.listdir(fixtures["corpus_dir"]) if f.endswith(".txt")]
    )
    assert len(kept_docs) + len(duplicate_records) + len(failure_records) == total_input_files

    failed_filenames = {record["filename"] for record in failure_records}
    assert fixtures["invalid_utf8_filename"] in failed_filenames
    assert fixtures["zero_byte_filename"] in failed_filenames

    kept_doc_ids = {doc["doc_id"] for doc in kept_docs}
    duplicate_doc_ids = {record["doc_id"] for record in duplicate_records}
    truncated_doc_id = doc_id_for_filename(fixtures["truncated_filename"])
    assert truncated_doc_id in kept_doc_ids or truncated_doc_id in duplicate_doc_ids


def test_exact_and_near_duplicates_are_caught(pg19_corpus_with_fixtures):
    fixtures = pg19_corpus_with_fixtures
    kept_docs, metadata_records, duplicate_records, failure_records = ingest_directory(
        fixtures["corpus_dir"]
    )

    kept_doc_ids = {doc["doc_id"] for doc in kept_docs}
    duplicate_doc_ids = {record["doc_id"] for record in duplicate_records}

    first_doc_id = doc_id_for_filename(fixtures["first_filename"])
    exact_duplicate_doc_id = doc_id_for_filename(fixtures["exact_duplicate_filename"])
    near_duplicate_doc_id = doc_id_for_filename(fixtures["near_duplicate_filename"])

    exact_family = {first_doc_id, exact_duplicate_doc_id}
    assert len(exact_family & kept_doc_ids) == 1
    assert len(exact_family & duplicate_doc_ids) == 1

    assert near_duplicate_doc_id in duplicate_doc_ids
    near_duplicate_record = next(
        record for record in duplicate_records if record["doc_id"] == near_duplicate_doc_id
    )
    assert near_duplicate_record["reason"] == "near"
    assert near_duplicate_record["duplicate_of_doc_id"] in exact_family


def test_every_surviving_doc_has_non_null_content_hash(pg19_corpus_with_fixtures):
    kept_docs, metadata_records, duplicate_records, failure_records = ingest_directory(
        pg19_corpus_with_fixtures["corpus_dir"]
    )

    assert len(metadata_records) == len(kept_docs)
    for metadata_record in metadata_records:
        assert metadata_record["content_hash"] is not None
        assert len(metadata_record["content_hash"]) == 64
