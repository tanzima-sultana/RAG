import argparse
import time
import torch
import sys
import pyarrow.parquet as pq

from constants import CPU, DENSE, BM25, HYBRID

from src.dist.dataset import Dataset
from src.dist.chunking import Chunking
from src.dist.embedding import Embedding

from src.indexing import Indexing
from src.eval_qa import EvalQA
from src.retrieval import Retrieval
from src.evaluation import Evaluation

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
    parser.add_argument("--num_eval_query", type=int, default=5,
                         help="Number of evaluation query")
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
    num_eval_query = args.num_eval_query
    k = args.k
    retrieval_type = args.retrieval_type
    reranking = args.reranking
    rerank_k = args.rerank_k

    if not torch.cuda.is_available():
        device = CPU

    # ----- 2. Load dataset
    print("\n ------- Load dataset -------- \n")
    s2 = time.time()

    df = Dataset(mode, dataset_size)
    dataset_path = df.load_parquet_dataset()

    if not dataset_path:
        print("Dataset failed, exiting")
        sys.exit(1)

    t2 = time.time() - s2
    print("time : ", t2)

    
    # -------------------- 3. Chunking
    print("\n ------- Chunking -------- \n")
    print("Type : ", chunking_type)

    s3 = time.time()

    ch = Chunking(num_partition, mode, model_name, dataset_path, dataset_size, device, chunking_type)
    chunk_path = ch.compute_chunks(max_chunk_size, fix_chunk_overlap, semantic_threshold)

    if not chunk_path:
        print("Chunking failed, exiting")
        sys.exit(1)

    t3 = time.time() - s3
    print("time : ", t3)

    
    # ----------- 4. Embedding 
    print("\n----- Embedding ------------\n")
    s4 = time.time()

    em = Embedding(num_partition, mode, model_name, dataset_size, device, chunking_type)
    embedding_path = em.generate_embeddings(chunk_path, batch_size)

    if not embedding_path:
        print("Embedding failed, exiting")
        sys.exit(1)

    t4 = time.time() - s4
    print("time : ", t4)

    # ----------- 5. Index 
    print("\n----- Indexing------------\n")
    s5 = time.time()

    idx = Indexing(mode, dataset_size, device, chunking_type, indexing_type)

    faiss_index = None
    if retrieval_type in (DENSE, HYBRID):
        print("FAISS Indexing : ", indexing_type)
        faiss_index = idx.generate_faiss_index(embedding_path)
    
    bm25_index = None 
    if retrieval_type in (BM25, HYBRID):
        print("Bm25 Indexing")
        bm25_index = idx.generate_bm25_index(chunk_path)

    if not faiss_index and not bm25_index: 
        print("Index failed, exiting")
        sys.exit(1)

    t5 = time.time() - s5
    print("time : ", t5)

    # ----------- 6. Evaluation Qus-Ans Set
    print("\n----- Evaluation Qus-Ans Set------------\n")
    s6 = time.time()

    ev = EvalQA(mode, dataset_size, device, chunking_type, num_eval_query)
    eval_set = ev.get_eval_set()

    if not eval_set:
        print("Eval set failed, exiting")
        sys.exit(1)

    t6 = time.time() - s6
    print("time : ", t6)

    # ----------- 7. Retrival
    print("\n----- Retrieval------------\n")
    print("Type : ", retrieval_type)
    s7 = time.time()

    # Read chunks
    table = pq.read_table(chunk_path)
    chunks = table.to_pylist()

    dry_run = True
    ret = Retrieval(dry_run, retrieval_type, chunks, eval_set, k, reranking, rerank_k, model_name, device)

    retrieved_output = None 
    if retrieval_type == DENSE:
        retrieved_output = ret.retrieval_dense(faiss_index)
    elif retrieval_type == BM25:
        retrieved_output = ret.retrieval_bm25(bm25_index)
    else:
        retrieved_output = ret.retrieval_hybrid(faiss_index, bm25_index)

    #print(retrieved_output)
    if not retrieved_output:
        print("Retrival failed, exiting")
        sys.exit(1)

    
    t7 = time.time() - s7
    print("time : ", t7)

    # ----------- 8. Evaluation 
    print("\n----- Evaluation ------------\n")
    s8 = time.time()

    use_faithfulness=False
    use_relevancy=False
    use_llm_correctness=False

    eval = Evaluation(mode, dataset_size, device, chunking_type, retrieval_type, model_name)
    eval.evaluate(k, retrieved_output, use_faithfulness, use_relevancy, use_llm_correctness)
    
    t8 = time.time() - s8
    print("time : ", t8)

    # --------------------
    t1 = time.time() - s1
    print("\n----- Total time : ", t1)

    