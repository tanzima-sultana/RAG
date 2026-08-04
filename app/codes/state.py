
import faiss
import pickle
import json 
import sys 
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import S3_BUCKET
from constants import LOCAL, AWS, FIXED, SENTENCE, SEMANTIC, DENSE, BM25, HYBRID, INDEX_FLATIP, INDEX_IVF, INDEX_HNSW

CHUNKS_MAP = None
FAISS_INDEX = None
FAISS_IDS = None
BM25_INDEX = None
BM25_IDS = None
QDRANT_NAME = None 
EVAL_SET = None
MODEL = None
CROSS_ENCODER = None

def load_state(mode, model_name, dataset_size, chunking_type, index_type):
    global CHUNKS_MAP, FAISS_INDEX, FAISS_IDS, BM25_INDEX, BM25_IDS, QDRANT_NAME, EVAL_SET, MODEL, CROSS_ENCODER
    
    manifest_path = f"manifests/{dataset_size}_manifest.json"
    if mode == AWS:
        manifest_path = f"s3://{S3_BUCKET}/" + manifest_path
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # 1. Chunks
    CHUNKS_MAP = None 
    with open(manifest[chunking_type]["chunk_path"], "rb") as f:
        CHUNKS_MAP = pickle.load(f)
    
    if not CHUNKS_MAP:
        print("Load chunks_map failed, exiting")
        sys.exit(1)

     # 2. FAISS-Index
    FAISS_INDEX = faiss.read_index(manifest[chunking_type][index_type])

    FAISS_IDS = None 
    with open(manifest[chunking_type]["chunk_ids"], "rb") as f:
        FAISS_IDS = pickle.load(f)
    
    if not FAISS_INDEX or not FAISS_IDS:
        print("Load faiss failed, exiting")
        sys.exit(1)
    
    # 3. BM25-Index
    BM25_INDEX = None 
    with open(manifest[chunking_type][BM25] , "rb") as f:
        BM25_INDEX = pickle.load(f)

    BM25_IDS = None 
    with open(manifest[chunking_type]["bm25_ids"] , "rb") as f:
        BM25_IDS = pickle.load(f)
    
    if not BM25_INDEX or not BM25_IDS:
        print("Load faiss failed, exiting")
        sys.exit(1)
    
    # 4. Qdrant name
    QDRANT_NAME = manifest[chunking_type]["vectordb"] 

    # 5. Evaluation
    with open(manifest[chunking_type]["eval_path"], "r") as f:
        EVAL_SET = json.load(f)
    
    # 6. Model

    MODEL = SentenceTransformer(model_name, device="cuda")
    CROSS_ENCODER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') 
