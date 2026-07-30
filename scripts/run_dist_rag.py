import argparse
import time
import torch
import sys
import pyarrow.parquet as pq

from constants import CPU, DENSE, BM25, HYBRID

from src.dataset import Dataset
from src.dist.chunking_embedding import Chunking_Embedding
from src.indexing import Indexing

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed RAG pipeline")

    parser.add_argument("--mode", type=str, required=True,
                         choices=["local", "aws"],
                         help="Local or distributed mode")
    parser.add_argument("--num_partition", type=int, default=4,
                         help="Number of partitions")
    parser.add_argument("--device", type=str, required=True,
                         choices=["cpu", "cuda"],
                         help="CPU or GPU")
    parser.add_argument("--model_name", type=str, required=True,
                         help="Transformer model name")
    parser.add_argument("--dataset_size", type=int, required=True,
                         help="Number of documents to process")
    parser.add_argument("--batch_size", type=int, required=True,
                         help="Number of batch to process")
    parser.add_argument("--chunking_type", type=str, required=True,
                         choices=["fixed", "sentence", "semantic"],
                         help="Chunking strategy to use")
    parser.add_argument("--max_chunk_size", type=int, default=256,
                         help="Max tokens per chunk (fixed/sentence chunking)")
    parser.add_argument("--fix_chunk_overlap", type=int, default=32,
                         help="Token overlap for fixed chunking")
    parser.add_argument("--semantic_threshold", type=float, default=0.5,
                         help="Cosine similarity threshold for semantic chunking split point")
    parser.add_argument("--indexing_type", type=str, required=True,
                         choices=["flatip", "ivf", "hnsw"],
                         help="Indexing type to use")

    return parser.parse_args()



if __name__ == "__main__":

    print("\n ------------- Dist RAG -------- \n")
    s1 = time.time()

    # ---------------- 1. args
    args = parse_args()
    print(args)

    mode = args.mode
    num_partition = args.num_partition
    device = args.device
    model_name = args.model_name
    dataset_size = args.dataset_size
    batch_size = args.batch_size
    chunking_type = args.chunking_type
    max_chunk_size = args.max_chunk_size
    fix_chunk_overlap = args.fix_chunk_overlap
    semantic_threshold = args.semantic_threshold
    indexing_type = args.indexing_type

    if not torch.cuda.is_available():
        device = CPU

    # ----- 2. Load dataset
    print("\n ------- Load dataset -------- \n")
    s2 = time.time()

    df = Dataset(mode, dataset_size)
    #dataset_path = df.load_parquet_dataset_s3()
    dataset_path = df.load_parquet_dataset_local()

    if not dataset_path:
        print("Dataset failed, exiting")
        sys.exit(1)

    t2 = time.time() - s2
    print("time : ", t2)

    
    # -------------------- 3. Chunking & Embedding
    print("\n ------- Chunking & Embedding-------- \n")
    print("Type : ", chunking_type)

    s3 = time.time()

    ch_em = Chunking_Embedding(mode, num_partition, model_name, dataset_path, dataset_size, device, chunking_type, batch_size)
    chunk_path, embedding_path = ch_em.compute_chunks_embeddings(max_chunk_size, fix_chunk_overlap, semantic_threshold)

    if not chunk_path or not embedding_path:
        print("Chunking or embedding failed, exiting")
        sys.exit(1)

    t3 = time.time() - s3
    print("time : ", t3)

    '''
    # ----------- 4. Index 
    print("\n----- Indexing------------\n")
    s4 = time.time()

    idx = Indexing(mode, dataset_size, device, chunking_type, indexing_type)

    print("FAISS Indexing : ", indexing_type)
    faiss_index = idx.generate_faiss_index(embedding_path)
    
    print("Bm25 Indexing")
    bm25_index = idx.generate_bm25_index(chunk_path)

    if not faiss_index and not bm25_index: 
        print("Index failed, exiting")
        sys.exit(1)

    t4 = time.time() - s4
    print("time : ", t4)

    # --------------------
    t1 = time.time() - s1
    print("\n----- Total time : ", t1)
    '''

    