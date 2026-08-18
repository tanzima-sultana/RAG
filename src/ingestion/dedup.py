import hashlib

NEAR_DUPLICATE_SHINGLE_SIZE = 5
NEAR_DUPLICATE_SIMILARITY_THRESHOLD = 0.8


def compute_content_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_word_shingles(text, shingle_size):
    words = text.split()
    shingles = set()
    for start_index in range(len(words) - shingle_size + 1):
        shingle = " ".join(words[start_index:start_index + shingle_size])
        shingles.add(shingle)
    return shingles


def jaccard_similarity(shingles_a, shingles_b):
    if len(shingles_a) == 0 or len(shingles_b) == 0:
        return 0.0
    intersection_size = len(shingles_a & shingles_b)
    union_size = len(shingles_a | shingles_b)
    return intersection_size / union_size


def deduplicate(docs):
    kept_docs = []
    kept_doc_hashes = []
    kept_doc_shingles = []
    duplicate_records = []

    for doc in docs:
        content_hash = compute_content_hash(doc["text"])
        shingles = build_word_shingles(doc["text"], NEAR_DUPLICATE_SHINGLE_SIZE)

        duplicate_of_doc_id = None
        duplicate_reason = None

        for kept_index in range(len(kept_docs)):
            if content_hash == kept_doc_hashes[kept_index]:
                duplicate_of_doc_id = kept_docs[kept_index]["doc_id"]
                duplicate_reason = "exact"
                break

            similarity = jaccard_similarity(shingles, kept_doc_shingles[kept_index])
            if similarity >= NEAR_DUPLICATE_SIMILARITY_THRESHOLD:
                duplicate_of_doc_id = kept_docs[kept_index]["doc_id"]
                duplicate_reason = "near"
                break

        if duplicate_of_doc_id is not None:
            duplicate_records.append({
                "doc_id": doc["doc_id"],
                "duplicate_of_doc_id": duplicate_of_doc_id,
                "reason": duplicate_reason,
            })
        else:
            kept_docs.append(doc)
            kept_doc_hashes.append(content_hash)
            kept_doc_shingles.append(shingles)

    return kept_docs, duplicate_records
