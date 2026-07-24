import argparse
import time
import torch

from constants import LOCAL, CPU

from src.dataset import Dataset
from src.chunking import Chunking


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed RAG pipeline")

    parser.add_argument("--mode", type=str, required=True,
                         choices=["local", "aws"],
                         help="Local or distributed mode")
    parser.add_argument("--device", type=str, required=True,
                         choices=["cpu", "cuda"],
                         help="CPU or GPU")
    parser.add_argument("--dataset_size", type=int, required=True,
                         help="Number of documents to process")
    parser.add_argument("--chunking_type", type=str, required=True,
                         choices=["fixed", "sentence", "semantic"],
                         help="Chunking strategy to use")
    parser.add_argument("--max_chunk_size", type=int, default=256,
                         help="Max tokens per chunk (fixed/sentence chunking)")
    parser.add_argument("--fix_chunk_overlap", type=int, default=32,
                         help="Token overlap for fixed chunking")
    parser.add_argument("--semantic_threshold", type=float, default=0.5,
                         help="Cosine similarity threshold for semantic chunking split point")
    parser.add_argument("--k", type=int, default=5,
                         help="Number of chunks to retrieve for eval")
    parser.add_argument("--retrieval_type", type=str, required=True,
                         choices=["dense", "bm25", "hybrid"],
                         help="Retrieval method")
    parser.add_argument("--reranking", type=int, default=0,
                         choices=[0, 1],
                         help="0 = no rerank, 1 = rerank")
    parser.add_argument("--rerank_k", type=int, default=5,
                         help="Number of chunks chosen/passed to the LLM after retrieval/reranking")

    return parser.parse_args()



if __name__ == "__main__":

    print("\n ------------- Dist RAG ----------- \n")
    s1 = time.time()

    # ---------------- 1. args
    args = parse_args()
    print(args)

    mode = args.mode
    device = args.device
    dataset_size = args.dataset_size
    chunking_type = args.chunking_type
    max_chunk_size = args.max_chunk_size
    fix_chunk_overlap = args.fix_chunk_overlap
    semantic_threshold = args.semantic_threshold
    k = args.k
    retrieval_type = args.retrieval_type
    reranking = args.reranking
    rerank_k = args.rerank_k

    if not torch.cuda.is_available():
        device = CPU

    # ----- 2. Load dataset
    s2 = time.time()

    df = Dataset(dataset_size)
    dataset = None

    if mode == LOCAL:
        dataset = df.load_parquet_dataset()
    else:
        dataset = df.load_parquet_dataset_s3()

    t2 = time.time() - s2
    print("\n----- Dataset load time : ", t2)

    # -------------------- 3. Chunking

    s3 = time.time()

    ch = Chunking(dataset, dataset_size, mode, device, chunking_type)
    chunks, avg_chunk_size = ch.compute_chunks(max_chunk_size, fix_chunk_overlap, semantic_threshold)

    t3 = time.time() - s3
    print("\n----- Chunking time : ", t3)


    # --------------------
    t1 = time.time() - s1
    print("\n----- Total time : ", t1)