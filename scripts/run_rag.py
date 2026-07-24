import argparse
import time
import torch

from constants import LOCAL, CPU, DENSE, BM25, HYBRID

from src.local.dataset import Dataset
from src.local.chunking import Chunking
from src.local.embedding import Embedding
from src.local.indexing import Indexing
from src.local.eval_qa import EvalQA
from src.local.retrieval import Retrieval
from src.local.evaluation import Evaluation

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed RAG pipeline")

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
    parser.add_argument("--no_eval_query", type=int, default=5,
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

    print("\n ------------- Local RAG -------- \n")
    s1 = time.time()

    # ---------------- 1. args
    args = parse_args()
    print(args)

    mode = args.mode
    device = args.device
    model_name = args.model_name
    dataset_size = args.dataset_size
    chunking_type = args.chunking_type
    max_chunk_size = args.max_chunk_size
    fix_chunk_overlap = args.fix_chunk_overlap
    semantic_threshold = args.semantic_threshold
    indexing_type = args.indexing_type
    no_eval_query = args.no_eval_query
    k = args.k
    retrieval_type = args.retrieval_type
    reranking = args.reranking
    rerank_k = args.rerank_k

    if not torch.cuda.is_available():
        device = CPU

    # ----- 2. Load dataset
    print("\n ------- Load dataset -------- \n")
    s2 = time.time()

    df = Dataset(dataset_size)
    dataset = None
    dataset = df.load_parquet_dataset()

    t2 = time.time() - s2
    print("time : ", t2)

    # -------------------- 3. Chunking
    print("\n ------- Chunking -------- \n")
    print("Type : ", chunking_type)

    s3 = time.time()

    ch = Chunking(model_name, dataset, dataset_size, device, chunking_type)
    chunks, avg_chunk_size = ch.compute_chunks(max_chunk_size, fix_chunk_overlap, semantic_threshold)

    t3 = time.time() - s3
    print("time : ", t3)

    # ----------- 4. Embedding 
    print("\n----- Embedding ------------\n")
    s4 = time.time()

    em = Embedding(model_name, dataset_size, device, chunking_type)
    embeddings = em.generate_embeddings(chunks)

    t4 = time.time() - s4
    print("time : ", t4)


    # ----------- 5. Index 
    print("\n----- Indexing------------\n")
    s5 = time.time()

    idx = Indexing(dataset_size, device, chunking_type, indexing_type)

    faiss_index = None
    if retrieval_type in (DENSE, HYBRID):
        print("FAISS Indexing : ", indexing_type)
        faiss_index = idx.generate_faiss_index(embeddings)
    
    bm25_index = None 
    if retrieval_type in (BM25, HYBRID):
        print("Bm25 Indexing")
        bm25_index = idx.generate_bm25_index(chunks)

    t5 = time.time() - s5
    print("time : ", t5)

    # ----------- 6. Evaluation Qus-Ans Set
    print("\n----- Evaluation Qus-Ans Set------------\n")
    s6 = time.time()

    ev = EvalQA(dataset_size, device, chunking_type, no_eval_query)
    eval_set = ev.build_eval_set(chunks, min_chunk_size=100)

    t6 = time.time() - s6
    print("time : ", t6)

    # ----------- 7. Retrival
    print("\n----- Retrieval------------\n")
    print("Type : ", retrieval_type)
    s7 = time.time()

    dry_run = True
    ret = Retrieval(dry_run, retrieval_type, chunks, eval_set, k, reranking, rerank_k, model_name, device)

    retrieved_output = None 
    if retrieval_type == DENSE:
        retrieved_output = ret.retrieval_dense(faiss_index)
    elif retrieval_type == BM25:
        retrieved_output = ret.retrieval_bm25(bm25_index)
    else:
        retrieved_output = ret.retrieval_hybrid(faiss_index, bm25_index)

    print(retrieved_output)
    t7 = time.time() - s7
    print("time : ", t7)

    # ----------- 8. Evaluation 
    print("\n----- Evaluation ------------\n")
    s8 = time.time()

    use_faithfulness=False
    use_relevancy=False
    use_llm_correctness=False

    eval = Evaluation(dataset_size, device, chunking_type, retrieval_type, model_name)
    eval_summary = eval.evaluate(k, retrieved_output, use_faithfulness, use_relevancy, use_llm_correctness)
    print(eval_summary)
    
    t8 = time.time() - s8
    print("time : ", t8)


    # --------------------
    t1 = time.time() - s1
    print("\n----- Total time : ", t1)