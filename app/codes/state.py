
import faiss
import pickle
import json 

from config import S3_BUCKET
from constants import LOCAL, AWS

CHUNKS_MAP = None
FAISS_INDEX = None
FAISS_IDS = None
BM25_INDEX = None
BM25_IDS = None
QDRANT_NAME = None 

def load_state(mode, dataset_size, chunking_type, index_type):
    global CHUNKS_MAP, FAISS_INDEX, FAISS_IDS, BM25_INDEX, BM25_IDS, QDRANT_NAME
    
    manifest_path = f"manifests/{dataset_size}_manifest.json"
    if mode == AWS:
        manifest_path = f"s3://{S3_BUCKET}/" + manifest_path
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    chunk_path = manifest[chunking_type]["chunk_path"]

    faiss_path = manifest[chunking_type][index_type]
    faiss_ids_path = manifest[chunking_type]["chunk_ids"]

    bm25_path = manifest[chunking_type]["bm25"]
    bm25_ids_path = manifest[chunking_type]["bm25_ids"]

    # Load chunks
    with open(chunk_path, "rb") as f:
        CHUNKS_MAP = pickle.load(f)

    # Load index
    FAISS_INDEX = faiss.read_index(faiss_path)
    with open(faiss_ids_path, "rb") as f:
        FAISS_IDS = pickle.load(f)
    with open(bm25_path, "rb") as f:
        BM25_INDEX = pickle.load(f)
    with open(bm25_ids_path, "rb") as f:
        BM25_IDS = pickle.load(f)
    
    # Qdrant
    QDRANT_NAME = manifest[chunking_type]["vectordb"] 
