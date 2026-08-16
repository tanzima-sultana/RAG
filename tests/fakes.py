"""Lightweight stand-ins for the heavy/network-bound dependencies (sentence-transformers
models, the Qdrant client, the Anthropic client) so the RAG pipeline's own logic can be
unit tested offline, deterministically, and fast.
"""
import hashlib

import numpy as np

EMBED_DIM = 16


def embed_text(text):
    vec = np.zeros(EMBED_DIM, dtype="float32")
    for word in text.lower().split():
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % EMBED_DIM] += 1.0
    if not vec.any():
        vec[0] = 1.0
    return vec


class FakeTokenizer:
    """Word-level stand-in for a HuggingFace tokenizer: one token per whitespace-split word."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


class FakeSentenceTransformer:
    def __init__(self, model_name=None, device=None):
        self.model_name = model_name
        self.device = device
        self.tokenizer = FakeTokenizer()

    def encode(self, sentences, batch_size=None, show_progress_bar=None, normalize_embeddings=False):
        single = isinstance(sentences, str)
        items = [sentences] if single else list(sentences)
        vectors = np.stack([embed_text(s) for s in items]).astype("float32")
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
        return vectors[0] if single else vectors


class FakeCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            a, b = embed_text(query), embed_text(text)
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
            scores.append(float(np.dot(a, b) / denom))
        return np.array(scores)


class _FakePoint:
    def __init__(self, point_id, vector, payload):
        self.id = point_id
        self.vector = vector
        self.payload = payload


class _FakeCollectionInfo:
    def __init__(self, points_count):
        self.points_count = points_count


class _FakeQueryResult:
    def __init__(self, points):
        self.points = points


class FakeQdrantClient:
    """In-memory stand-in for qdrant_client.QdrantClient covering the subset of the
    API this repo uses: collection lifecycle, upsert, and query_points."""

    def __init__(self, *args, **kwargs):
        self.collections = {}

    def collection_exists(self, collection_name):
        return collection_name in self.collections

    def create_collection(self, collection_name, vectors_config):
        self.collections[collection_name] = {"vectors_config": vectors_config, "points": {}}

    def delete_collection(self, collection_name):
        self.collections.pop(collection_name, None)

    def upsert(self, collection_name, points):
        for point in points:
            self.collections[collection_name]["points"][point.id] = point

    def get_collection(self, collection_name):
        return _FakeCollectionInfo(len(self.collections[collection_name]["points"]))

    def query_points(self, collection_name, query, limit, with_payload=True):
        points = list(self.collections.get(collection_name, {}).get("points", {}).values())
        return _FakeQueryResult(points[:limit])

    def set_hits(self, collection_name, chunk_ids):
        """Test helper: seed a collection with hits carrying the given chunk_ids, in order."""
        self.collections[collection_name] = {
            "vectors_config": None,
            "points": {i: _FakePoint(i, None, {"chunk_id": cid}) for i, cid in enumerate(chunk_ids)},
        }


class FakeAnthropicResponseUsage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeAnthropicContentBlock:
    def __init__(self, text):
        self.text = text


class FakeAnthropicMessage:
    def __init__(self, text, input_tokens, output_tokens):
        self.content = [FakeAnthropicContentBlock(text)]
        self.usage = FakeAnthropicResponseUsage(input_tokens, output_tokens)


class FakeAnthropicMessages:
    def __init__(self, text="MOCK_RESPONSE", input_tokens=10, output_tokens=5):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls = []

    def create(self, model, max_tokens, messages):
        self.calls.append({"model": model, "max_tokens": max_tokens, "messages": messages})
        return FakeAnthropicMessage(self.text, self.input_tokens, self.output_tokens)


class FakeAnthropicClient:
    def __init__(self, text="MOCK_RESPONSE", input_tokens=10, output_tokens=5, api_key=None):
        self.messages = FakeAnthropicMessages(text, input_tokens, output_tokens)
