
from rank_bm25 import BM25Okapi

def tokenize_chunk_text(text):
    return text.lower().split()

def generate_bm25_indexing(chunks):

    tokens = [tokenize_chunk_text(chunk['chunk_text']) for chunk in chunks]
    bm25_scores = BM25Okapi(tokens)
    return bm25_scores
