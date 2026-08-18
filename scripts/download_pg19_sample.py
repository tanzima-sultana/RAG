import argparse
import csv
import io
import json
import os
import re

import requests

TRAIN_FILES_LIST_URL = "https://huggingface.co/datasets/deepmind/pg19/resolve/main/data/train_files.txt"
METADATA_URL = "https://storage.googleapis.com/deepmind-gutenberg/metadata.csv"
ASSET_ROOT_URL = "https://storage.googleapis.com/deepmind-gutenberg/"


def parse_args():
    parser = argparse.ArgumentParser(description="Download a small pg19 sample as .txt files")

    parser.add_argument("--num_books", type=int, default=25,
                         help="Number of books to download from the pg19 train split")
    parser.add_argument("--output_dir", type=str, default="data/raw/pg19_sample",
                         help="Directory to write the sampled .txt files into")

    return parser.parse_args()


def sanitize_title(short_book_title):
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", short_book_title).strip("_")
    return sanitized.lower()


def build_filename(short_book_title, book_id):
    sanitized_title = sanitize_title(short_book_title)
    return f"{sanitized_title}_{book_id}.txt"


def fetch_first_train_file_paths(num_books):
    response = requests.get(TRAIN_FILES_LIST_URL, timeout=30)
    response.raise_for_status()
    all_train_file_paths = sorted(response.text.splitlines())
    return all_train_file_paths[:num_books]


def fetch_book_id_to_metadata():
    response = requests.get(METADATA_URL, timeout=60)
    response.raise_for_status()
    reader = csv.DictReader(
        io.StringIO(response.text),
        fieldnames=["book_id", "short_book_title", "publication_date", "url"],
    )
    return {row["book_id"]: row for row in reader}


def download_pg19_sample(num_books, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    train_file_paths = fetch_first_train_file_paths(num_books)
    book_id_to_metadata = fetch_book_id_to_metadata()

    filename_to_sidecar_metadata = {}
    num_books_downloaded = 0
    for train_file_path in train_file_paths:
        book_id = os.path.splitext(os.path.basename(train_file_path))[0]
        book_metadata = book_id_to_metadata[book_id]
        short_book_title = book_metadata["short_book_title"]

        text_response = requests.get(ASSET_ROOT_URL + train_file_path, timeout=60)
        text_response.raise_for_status()

        filename = build_filename(short_book_title, book_id)
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(text_response.text)

        filename_to_sidecar_metadata[filename] = {
            "book_id": book_id,
            "title": short_book_title,
            "url": book_metadata["url"],
            "publication_date": book_metadata["publication_date"],
        }

        print(f"Wrote {output_path}")
        num_books_downloaded += 1

    sidecar_path = os.path.join(output_dir, "metadata.json")
    with open(sidecar_path, "w", encoding="utf-8") as sidecar_file:
        json.dump(filename_to_sidecar_metadata, sidecar_file, indent=2)

    print(f"Downloaded {num_books_downloaded} books to {output_dir}")
    print(f"Wrote sidecar metadata to {sidecar_path}")


if __name__ == "__main__":
    args = parse_args()
    download_pg19_sample(args.num_books, args.output_dir)
