from datetime import datetime, timezone

from src.ingestion.dedup import compute_content_hash


def build_metadata(doc):
    return {
        "doc_id": doc["doc_id"],
        "title": doc["title"],
        "url": doc.get("url"),
        "source_format": doc["source_format"],
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
        "content_hash": compute_content_hash(doc["text"]),
    }
