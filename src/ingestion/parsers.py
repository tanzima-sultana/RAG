import json
import os
import re

import pypdf

import constants

PRODUCED_BY_LINE_PATTERN = re.compile(r"(?im)^[ \t]*Produced by .*$")
END_OF_GUTENBERG_LINE_PATTERN = re.compile(r"(?im)^[ \t]*End of (the )?Project Gutenberg.*$")
TRAILING_ASTERISKS_LINE_PATTERN = re.compile(r"(?m)^[ \t]*\*+[ \t]*$")

ARXIV_WATERMARK_LINE_PATTERN = re.compile(r"(?m)^[ \t]*arXiv:\d+\.\d+v\d+[ \t]*\[[\w.\-]+\][ \t]*\d{1,2}[ \t]+\w+[ \t]+\d{4}[ \t]*$")
PAGE_NUMBER_ONLY_LINE_PATTERN = re.compile(r"(?m)^[ \t]*\d{1,4}[ \t]*$")
MINIMUM_PAGE_COUNT_FOR_RUNNING_HEADER_DETECTION = 3


def strip_pg19_boilerplate(text):
    text_without_produced_by_line = PRODUCED_BY_LINE_PATTERN.sub("", text)
    text_without_end_of_gutenberg_line = END_OF_GUTENBERG_LINE_PATTERN.sub("", text_without_produced_by_line)
    text_without_trailing_asterisks = TRAILING_ASTERISKS_LINE_PATTERN.sub("", text_without_end_of_gutenberg_line)
    return text_without_trailing_asterisks.strip()


def load_sidecar_metadata(directory_path):
    sidecar_path = os.path.join(directory_path, "metadata.json")
    if not os.path.isfile(sidecar_path):
        return {}
    with open(sidecar_path, "r", encoding="utf-8") as sidecar_file:
        return json.load(sidecar_file)


def derive_title_from_filename(filename):
    name_without_extension = os.path.splitext(filename)[0]
    return name_without_extension.replace("_", " ").strip().title()


def source_format_for_extension(file_extension):
    if file_extension == ".txt":
        return constants.TXT
    if file_extension == ".md":
        return constants.MD
    if file_extension == ".pdf":
        return constants.PDF
    raise ValueError(f"Unsupported file format: {file_extension}")


def resolve_title_and_url(directory_path, filename):
    sidecar_metadata = load_sidecar_metadata(directory_path)
    file_sidecar_metadata = sidecar_metadata.get(filename, {})

    title = file_sidecar_metadata.get("title", derive_title_from_filename(filename))
    url = file_sidecar_metadata.get("url")

    return title, url


def parse_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as text_file:
        raw_text = text_file.read()

    cleaned_text = strip_pg19_boilerplate(raw_text)

    directory_path = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    doc_id = os.path.splitext(filename)[0]
    file_extension = os.path.splitext(filename)[1].lower()

    title, url = resolve_title_and_url(directory_path, filename)

    return {
        "doc_id": doc_id,
        "title": title,
        "url": url,
        "text": cleaned_text,
        "source_format": source_format_for_extension(file_extension),
    }


def extract_pdf_page_texts(file_path):
    reader = pypdf.PdfReader(file_path)
    page_texts = []
    for page in reader.pages:
        page_texts.append(page.extract_text())
    return page_texts


def strip_repeated_page_lines(page_texts):
    if len(page_texts) < MINIMUM_PAGE_COUNT_FOR_RUNNING_HEADER_DETECTION:
        return page_texts

    line_page_counts = {}
    for page_text in page_texts:
        lines_on_this_page = set(line.strip() for line in page_text.split("\n") if line.strip())
        for line in lines_on_this_page:
            line_page_counts[line] = line_page_counts.get(line, 0) + 1

    majority_page_count = len(page_texts) / 2
    repeated_lines = set()
    for line, page_count in line_page_counts.items():
        if page_count > majority_page_count:
            repeated_lines.add(line)

    page_texts_without_repeated_lines = []
    for page_text in page_texts:
        kept_lines = []
        for line in page_text.split("\n"):
            if line.strip() not in repeated_lines:
                kept_lines.append(line)
        page_texts_without_repeated_lines.append("\n".join(kept_lines))

    return page_texts_without_repeated_lines


def strip_pdf_boilerplate(text):
    text_without_arxiv_watermark = ARXIV_WATERMARK_LINE_PATTERN.sub("", text)
    text_without_page_numbers = PAGE_NUMBER_ONLY_LINE_PATTERN.sub("", text_without_arxiv_watermark)
    return text_without_page_numbers


def parse_pdf_file(file_path):
    page_texts = extract_pdf_page_texts(file_path)
    page_texts_without_running_headers = strip_repeated_page_lines(page_texts)

    cleaned_page_texts = []
    for page_text in page_texts_without_running_headers:
        cleaned_page_texts.append(strip_pdf_boilerplate(page_text))

    full_text = "\n".join(cleaned_page_texts).strip()

    directory_path = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    doc_id = os.path.splitext(filename)[0]
    file_extension = os.path.splitext(filename)[1].lower()

    title, url = resolve_title_and_url(directory_path, filename)

    return {
        "doc_id": doc_id,
        "title": title,
        "url": url,
        "text": full_text,
        "source_format": source_format_for_extension(file_extension),
    }


def parse_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".txt" or file_extension == ".md":
        return parse_text_file(file_path)

    if file_extension == ".pdf":
        return parse_pdf_file(file_path)

    raise ValueError(f"Unsupported file format: {file_extension}")
