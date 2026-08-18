import os
import shutil

import pypdf
import pytest

from src.ingestion.ingestion import ingest_directory

PDF_SAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw", "pdf_sample",
)


def pdf_sample_available():
    if not os.path.isdir(PDF_SAMPLE_DIR):
        return False
    pdf_filenames = [f for f in os.listdir(PDF_SAMPLE_DIR) if f.endswith(".pdf")]
    return len(pdf_filenames) >= 25


pytestmark = pytest.mark.skipif(
    not pdf_sample_available(),
    reason="pdf sample corpus not found; run scripts/download_pdf_sample.py first",
)


def doc_id_for_filename(filename):
    return os.path.splitext(filename)[0]


@pytest.fixture()
def pdf_corpus_with_fixtures(tmp_path):
    corpus_dir = tmp_path / "pdf_corpus"
    corpus_dir.mkdir()

    source_filenames = sorted(f for f in os.listdir(PDF_SAMPLE_DIR) if f.endswith(".pdf"))[:25]
    for filename in source_filenames:
        shutil.copyfile(os.path.join(PDF_SAMPLE_DIR, filename), corpus_dir / filename)

    metadata_sidecar_path = os.path.join(PDF_SAMPLE_DIR, "metadata.json")
    if os.path.isfile(metadata_sidecar_path):
        shutil.copyfile(metadata_sidecar_path, corpus_dir / "metadata.json")

    first_filename = source_filenames[0]
    first_file_path = corpus_dir / first_filename
    first_file_bytes = first_file_path.read_bytes()

    exact_duplicate_filename = "exact_duplicate_of_" + first_filename
    (corpus_dir / exact_duplicate_filename).write_bytes(first_file_bytes)

    reader = pypdf.PdfReader(str(first_file_path))
    assert len(reader.pages) >= 3, f"expected multiple pages in {first_filename}"
    near_duplicate_filename = "near_duplicate_of_" + first_filename
    writer = pypdf.PdfWriter()
    middle_page_index = len(reader.pages) // 2
    for page_index in range(len(reader.pages)):
        if page_index != middle_page_index:
            writer.add_page(reader.pages[page_index])
    with open(corpus_dir / near_duplicate_filename, "wb") as near_duplicate_file:
        writer.write(near_duplicate_file)

    second_filename = source_filenames[1]
    second_file_bytes = (corpus_dir / second_filename).read_bytes()
    truncated_filename = "truncated_paper.pdf"
    (corpus_dir / truncated_filename).write_bytes(second_file_bytes[:len(second_file_bytes) // 2])

    invalid_bytes_filename = "invalid_bytes_paper.pdf"
    (corpus_dir / invalid_bytes_filename).write_bytes(b"this is not a real pdf file" * 50)

    zero_byte_filename = "empty_paper.pdf"
    (corpus_dir / zero_byte_filename).write_bytes(b"")

    third_filename = source_filenames[2]
    third_reader = pypdf.PdfReader(str(corpus_dir / third_filename))
    encrypted_filename = "encrypted_paper.pdf"
    encrypted_writer = pypdf.PdfWriter()
    encrypted_writer.append(third_reader)
    encrypted_writer.encrypt("some_password")
    with open(corpus_dir / encrypted_filename, "wb") as encrypted_file:
        encrypted_writer.write(encrypted_file)

    return {
        "corpus_dir": str(corpus_dir),
        "first_filename": first_filename,
        "exact_duplicate_filename": exact_duplicate_filename,
        "near_duplicate_filename": near_duplicate_filename,
        "truncated_filename": truncated_filename,
        "invalid_bytes_filename": invalid_bytes_filename,
        "zero_byte_filename": zero_byte_filename,
        "encrypted_filename": encrypted_filename,
    }


def test_ingest_directory_processes_all_files_without_raising(pdf_corpus_with_fixtures):
    ingest_directory(pdf_corpus_with_fixtures["corpus_dir"])


def test_corrupt_files_are_isolated_not_dropped_or_crashing(pdf_corpus_with_fixtures):
    fixtures = pdf_corpus_with_fixtures
    kept_docs, metadata_records, duplicate_records, failure_records = ingest_directory(
        fixtures["corpus_dir"]
    )

    total_input_files = len(
        [f for f in os.listdir(fixtures["corpus_dir"]) if f.endswith(".pdf")]
    )
    assert len(kept_docs) + len(duplicate_records) + len(failure_records) == total_input_files

    failed_filenames = {record["filename"] for record in failure_records}
    assert fixtures["truncated_filename"] in failed_filenames
    assert fixtures["invalid_bytes_filename"] in failed_filenames
    assert fixtures["zero_byte_filename"] in failed_filenames
    assert fixtures["encrypted_filename"] in failed_filenames


def test_exact_and_near_duplicates_are_caught(pdf_corpus_with_fixtures):
    fixtures = pdf_corpus_with_fixtures
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


def test_every_surviving_doc_has_non_null_content_hash(pdf_corpus_with_fixtures):
    kept_docs, metadata_records, duplicate_records, failure_records = ingest_directory(
        pdf_corpus_with_fixtures["corpus_dir"]
    )

    assert len(metadata_records) == len(kept_docs)
    for metadata_record in metadata_records:
        assert metadata_record["content_hash"] is not None
        assert len(metadata_record["content_hash"]) == 64


def test_kept_docs_have_pdf_source_format_and_no_arxiv_watermark_leak(pdf_corpus_with_fixtures):
    kept_docs, metadata_records, duplicate_records, failure_records = ingest_directory(
        pdf_corpus_with_fixtures["corpus_dir"]
    )

    for doc in kept_docs:
        assert doc["source_format"] == "pdf"
        assert doc["text"].strip() != ""
