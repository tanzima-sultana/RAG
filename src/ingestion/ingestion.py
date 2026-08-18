import os

from src.ingestion.parsers import parse_file
from src.ingestion.preprocessing import preprocess_doc
from src.ingestion.dedup import deduplicate
from src.ingestion.metadata import build_metadata

SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf")


def list_source_files(directory_path):
    all_filenames = sorted(os.listdir(directory_path))
    source_filenames = []
    for filename in all_filenames:
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension in SUPPORTED_EXTENSIONS:
            source_filenames.append(filename)
    return source_filenames


def parse_all_files(directory_path):
    parsed_docs = []
    failure_records = []

    for filename in list_source_files(directory_path):
        file_path = os.path.join(directory_path, filename)

        try:
            doc = parse_file(file_path)
        except Exception as parse_error:
            failure_records.append({
                "filename": filename,
                "reason": "parse_error",
                "detail": str(parse_error),
            })
            print(f"WARNING: failed to parse {file_path}: {parse_error}")
            continue

        if doc["text"].strip() == "":
            failure_records.append({
                "filename": filename,
                "reason": "empty_content",
                "detail": "parsed text is empty",
            })
            print(f"WARNING: skipping {file_path}: empty content")
            continue

        parsed_docs.append(doc)

    return parsed_docs, failure_records


def ingest_directory(directory_path):
    parsed_docs, failure_records = parse_all_files(directory_path)

    preprocessed_docs = []
    for doc in parsed_docs:
        preprocessed_docs.append(preprocess_doc(doc))

    kept_docs, duplicate_records = deduplicate(preprocessed_docs)

    metadata_records = []
    for doc in kept_docs:
        metadata_records.append(build_metadata(doc))

    return kept_docs, metadata_records, duplicate_records, failure_records
