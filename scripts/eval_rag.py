import argparse
import time
import json
import sys 
import pickle 
import faiss 
from sentence_transformers import SentenceTransformer, CrossEncoder
from constants import FIXED, SENTENCE, SEMANTIC, DENSE, BM25, HYBRID, INDEX_FLATIP, INDEX_IVF, INDEX_HNSW

from src.retrieval import Retrieval
from src.evaluation import Evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="RAG Eval")

    parser.add_argument("--mock_run", type=int, default=1,
                         help="mock run 0 or 1")
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
                         help="Type of chunks")
    parser.add_argument("--index_type", type=str, required=True,
                         choices=["flatip", "ivf", "hnsw"],
                         help="Type of retrieval")
    parser.add_argument("--retrieval_type", type=str, required=True,
                         choices=["dense", "bm25", "hybrid", "vectordb"],
                         help="Type of retrieval")
    parser.add_argument("--num_queries", type=int, default=5,
                         help="Num of eval questions")
    parser.add_argument("--k", type=int, default=3,
                         help="Top k to retrieve")
    parser.add_argument("--re_ranking", type=int, default=0,
                         help="Reranking yes(1) or no(0)")
    parser.add_argument("--rerank_k", type=int, default=10,
                         help="Top k for reranking")

    return parser.parse_args()

if __name__ == "__main__":

    print("\n ------------- Eval RAG -------- \n")
    s1 = time.time()

    # ---------------- 1. args
    args = parse_args()
    print(args)

    mock_run = args.mock_run
    mode = args.mode
    device = args.device
    model_name = args.model_name
    dataset_size = args.dataset_size
    chunking_type = args.chunking_type
    index_type = args.index_type
    retrieval_type = args.retrieval_type
    num_queries = args.num_queries
    k = args.k
    re_ranking = args.re_ranking
    rerank_k = args.rerank_k

    # ------------ 2. Load paths 
    manifest_path = f"manifests/{dataset_size}_manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # 1. Chunks
    chunks_map = None
    with open(manifest[chunking_type]["chunk_path"], "rb") as f:
        chunks_map = pickle.load(f)
    
    if not chunks_map:
        print("Load chunks_map failed, exiting")
        sys.exit(1)

    
    # 2. FAISS-Index
    faiss_index = faiss.read_index(manifest[chunking_type][index_type])

    faiss_ids = None 
    with open(manifest[chunking_type]["chunk_ids"], "rb") as f:
        faiss_ids = pickle.load(f)
    
    if not faiss_index or not faiss_ids:
        print("Load faiss failed, exiting")
        sys.exit(1)
    
    # 3. BM25-Index
    bm25_index = None 
    with open(manifest[chunking_type][BM25] , "rb") as f:
        bm25_index = pickle.load(f)

    bm25_ids = None 
    with open(manifest[chunking_type]["bm25_ids"] , "rb") as f:
        bm25_ids = pickle.load(f)
    
    if not bm25_index or not bm25_ids:
        print("Load faiss failed, exiting")
        sys.exit(1)
    
    # 4. Qdrant name
    qdrant_name = manifest[chunking_type]["vectordb"] 

    # 5. Eval set
    eval_set = None
    with open(manifest[chunking_type]["eval_path"], "r") as f:
        eval_set = json.load(f)
    
    if eval_set is None:    
        print("Load eval_set failed, exiting")
        sys.exit(1)

    # ----------- 4. Retrival
    print("\n----- Retrieval------------\n")
    print("Type : ", retrieval_type)
    s4 = time.time()

    model = SentenceTransformer(model_name, device=device)
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') 

    ret1 = Retrieval(mock_run, mode, chunking_type, chunks_map, eval_set, k, re_ranking, rerank_k, model, cross_encoder)
    retrieved_output = None

    if retrieval_type == DENSE:
        retrieved_output = ret1.retrieval_dense(faiss_index, faiss_ids)
    elif retrieval_type == BM25:
        retrieved_output = ret1.retrieval_bm25(bm25_index, bm25_ids)
    elif retrieval_type == HYBRID:
        retrieved_output = ret1.retrieval_hybrid(faiss_index, faiss_ids, bm25_index, bm25_ids)
    else:
        # Qdrant
        collection_name = qdrant_name
        retrieved_output = ret1.retrieval_qdrant(collection_name)

    t4 = time.time() - s4
    print("time : ", t4)

    # ----------- 8. Evaluation 
    print("\n----- Evaluation ------------\n")
    s8 = time.time()

    if mock_run == 1:
        use_llm_judge = False
    else:
        use_llm_judge = True
        
    eval = Evaluation(mode, dataset_size, model_name, use_llm_judge)
    eval_summary = eval.evaluate(k, retrieved_output)
    
    t8 = time.time() - s8
    print("time : ", t8)
    