import re

WHITESPACE_RUN_PATTERN = re.compile(r"[ \t]+")
BLANK_LINE_RUN_PATTERN = re.compile(r"\n{3,}")


def normalize_whitespace(text):
    text_with_single_spaces = WHITESPACE_RUN_PATTERN.sub(" ", text)
    text_with_collapsed_blank_lines = BLANK_LINE_RUN_PATTERN.sub("\n\n", text_with_single_spaces)
    return text_with_collapsed_blank_lines.strip()


def preprocess_doc(doc):
    return {
        "doc_id": doc["doc_id"],
        "title": doc["title"],
        "url": doc.get("url"),
        "text": normalize_whitespace(doc["text"]),
        "source_format": doc["source_format"],
    }
