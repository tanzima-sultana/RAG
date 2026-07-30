import argparse
import time
import torch
import sys 
import json
import os

from constants import LOCAL, CPU, DENSE, BM25, HYBRID
from constants import FIXED, SENTENCE, SEMANTIC

from src.dataset import Dataset

from src.local.chunking import Chunking
from src.local.embedding import Embedding

from src.indexing import Indexing
from src.vector_db import VectorDB



def parse_args():
    parser = argparse.ArgumentParser(description="RAG build")

    parser.add_argument("--mode", type=str, required=True,
                         choices=["local", "aws"],
                         help="Local or distributed mode")
    parser.add_argument("--device", type=str, required=True,
                         choices=["cpu", "cuda"],
                         help="CPU or GPU")
    parser.add_argument("--model_name", type=str, required=True,
                         help="Transformer model name")
    parser.add_argument("--dataset_size", type=int, required=True,
                         help="Number of documents to process")
    parser.add_argument("--max_chunk_size", type=int, default=256,
                         help="Max tokens per chunk (fixed/sentence chunking)")
    parser.add_argument("--fix_chunk_overlap", type=int, default=32,
                         help="Token overlap for fixed chunking")
    parser.add_argument("--semantic_threshold", type=float, default=0.5,
                         help="Cosine similarity threshold for semantic chunking split point")
    parser.add_argument("--ivf_nlist", type=int, default=256,
                         help="Number of IVF nlist")
    parser.add_argument("--hnsw_m", type=int, default=32,
                         help="Number of HNSW M")

    return parser.parse_args()



if __name__ == "__main__":

    print("\n ------------- RAG -------- \n")
    s1 = time.time()

    # ---------------- 1. args
    args = parse_args()
    print(args)

    mode = args.mode
    device = args.device
    model_name = args.model_name
    dataset_size = args.dataset_size
    max_chunk_size = args.max_chunk_size
    fix_chunk_overlap = args.fix_chunk_overlap
    semantic_threshold = args.semantic_threshold
    ivf_nlist = args.ivf_nlist
    hnsw_m = args.hnsw_m

    if not torch.cuda.is_available():
        device = CPU

    # ----- 2. Load dataset
    print("\n ------- Load dataset -------- \n")
    s2 = time.time()

    df = Dataset(mode, dataset_size)
    dataset = df.load_parquet_dataset_local()

    if not dataset:
        print("Dataset failed, exiting")
        sys.exit(1)

    t2 = time.time() - s2
    print("time : ", t2)

    # -------------------- 3. Chunking
    print("\n ------- Chunking -------- \n")

    s3 = time.time()

    ch = Chunking(dataset, dataset_size, device, model_name)
    fixed_chunks_path, sentence_chunks_path, semantic_chunks_path = ch.compute_chunks(max_chunk_size, fix_chunk_overlap, semantic_threshold)
    
    if not fixed_chunks_path or not sentence_chunks_path or not semantic_chunks_path:
        print("Chunking failed, exiting")
        sys.exit(1)

    t3 = time.time() - s3
    print("time : ", t3)

    
    # ----------- 4. Embedding 
    print("\n----- Embedding ------------\n")
    s4 = time.time()

    em = Embedding(dataset_size, device, model_name)

    embedding_fixed_path = em.generate_embeddings(fixed_chunks_path)
    embedding_sentence_path = em.generate_embeddings(sentence_chunks_path)
    embedding_semantic_path = em.generate_embeddings(semantic_chunks_path)

    if not embedding_fixed_path or not embedding_sentence_path or not embedding_semantic_path:
        print("Chunking failed, exiting")
        sys.exit(1)

    t4 = time.time() - s4
    print("time : ", t4)

    
    # ----------- 5. Index 
    print("\n----- Indexing------------\n")
    s5 = time.time()

    idx = Indexing(mode, dataset_size, device)

    # 1. FAISS 
    # Fixed, Sentence & Semantic
    flatip_path1, ivf_path1, hnsw_path1, chunk_ids_path1 = idx.generate_faiss_index(embedding_fixed_path, ivf_nlist, hnsw_m)
    flatip_path2, ivf_path2, hnsw_path2, chunk_ids_path2 = idx.generate_faiss_index(embedding_sentence_path, ivf_nlist, hnsw_m)
    flatip_path3, ivf_path3, hnsw_path3, chunk_ids_path3 = idx.generate_faiss_index(embedding_semantic_path, ivf_nlist, hnsw_m)

    # 2. BM25
    # Fixed, Sentence & Semantic
    bm25_path1, bm25_ids_path1 = idx.generate_bm25_index(fixed_chunks_path)
    bm25_path2, bm25_ids_path2 = idx.generate_bm25_index(sentence_chunks_path)
    bm25_path3, bm25_ids_path3 = idx.generate_bm25_index(semantic_chunks_path)

    t5 = time.time() - s5
    print("time : ", t5)

    # 3. Vector DB (qdarnt)
    db_name = "vectordb"
    vdb = VectorDB(db_name)

    # Fixed, Sentence & Semantic
    # Saving the names of the collection
    vectordb_fixed = vdb.create_vector_db(embedding_fixed_path)
    vectordb_sentence = vdb.create_vector_db(embedding_sentence_path)
    vectordb_semantic = vdb.create_vector_db(embedding_semantic_path)

    if not vectordb_fixed or not vectordb_sentence or not vectordb_semantic:
        print("VectorDB failed, exiting")
        sys.exit(1)

    t1 = time.time() - s1
    print("\nTotal time : ", t1)

    # Save index paths and vector DB collection name

    manifest = {
        "fixed": {
            "flatip": flatip_path1, "ivf": ivf_path1, "hnsw": hnsw_path1, "chunk_ids": chunk_ids_path1, 
            "bm25": bm25_path1, "bm25_ids": bm25_ids_path1,
            "vectordb_fixed": vectordb_fixed,
        },
        "sentence": {
            "flatip": flatip_path2, "ivf": ivf_path2, "hnsw": hnsw_path2, "chunk_ids": chunk_ids_path2, 
            "bm25": bm25_path2, "bm25_ids": bm25_ids_path2,
            "vectordb_sentence": vectordb_sentence,
        },
        "semantic": {
            "flatip": flatip_path3, "ivf": ivf_path3, "hnsw": hnsw_path3, "chunk_ids": chunk_ids_path3, 
            "bm25": bm25_path3, "bm25_ids": bm25_ids_path3,
            "vectordb_semantic": vectordb_semantic,
        },
    }

    os.makedirs("manifests", exist_ok=True)
    manifest_path = f"manifests/{dataset_size}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nManifest saved : ", manifest_path)

   