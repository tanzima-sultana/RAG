
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
import pickle

from constants import FIXED, SENTENCE, SEMANTIC, DENSE, BM25, HYBRID
from dataset import load_processed_dataset
from chunking import compute_chunks, fixed_chunking, sentence_aware_chunking, semantic_chunking
from embedding import generate_embeddings
from faiss_indexing import generate_FlatIndexIP
from bm25_indexing import generate_bm25_indexing
from retrieval import retrieve_chunks
from generate_eval_set import build_eval_set 
from evaluation import evaluate
from cost_calculation import estimate_cost

model = SentenceTransformer('all-MiniLM-L6-v2')

# ----- main
if __name__ == "__main__":  

    # arg 1 : Dataset size
    dataset_size = int(sys.argv[1])

    # arg 2 : Chunk type
    chunk_type = sys.argv[2] if len(sys.argv) > 2 else 'fixed'
    valid_chunk_types = [FIXED, SENTENCE, SEMANTIC]
    if chunk_type not in valid_chunk_types:
        raise ValueError(f"chunk_type must be one of {valid_chunk_types}, got '{chunk_type}'")
    
    # arg 3 : Retrieval type
    retrieval_type = sys.argv[3] if len(sys.argv) > 3 else 'dense'
    valid_retrieval_types = [DENSE, BM25, HYBRID]
    if retrieval_type not in valid_retrieval_types:
        raise ValueError(f"retrieval_type must be one of {valid_retrieval_types}, got '{retrieval_type}'")

    # arg 4 : k = 3 by default
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 3

    # ----- 1. Load dataset
    dataset = load_processed_dataset()
    n_samples = dataset_size
    sample = dataset.select(range(n_samples)) 

    print("Dataset size : ", dataset_size, ", chunk type : ", chunk_type, ", retrieval type : ", retrieval_type, ", k : ", k)

    # ----- 2. Chunking
    chunks = []
    avg_tokens = 0
    chunks, avg_tokens = compute_chunks(sample, n_samples, chunk_type)
    
    # ----- 3. Load eval dataset
    #eval_set = build_nq_eval_set(dataset, dataset_size, chunks)
    min_chunk_size = 100
    no_of_qus = 50
    eval_set = build_eval_set(chunk_type, dataset, dataset_size, chunks, min_chunk_size, no_of_qus)
    #print(len(eval_set), eval_set)

    # Cost estimation
    cost = 0
    cost = estimate_cost(len(eval_set), k, avg_tokens, use_faithfulness=True, use_relevancy=True, use_llm_correctness=True)
    
    # Indexing
    indexing = []

    # BM25 
    bm25 = []

    print("Retrieval type : ", retrieval_type)
    retrieved_output = []
    eval_metrices = []
    if retrieval_type == DENSE or retrieval_type == HYBRID:

        # ----- 4. Embedding
        embeddings = generate_embeddings(chunks, n_samples, chunk_type)

        # ----- 5. FAISS indexing

        indexing = generate_FlatIndexIP(embeddings, n_samples, chunk_type)

        # All of these maintain same indexing strategy, so we can use the same index for all of them. indexing_fixed[0] has the indexing for
        # embeddings_fixed[0] and embeddings_fixed[0] has the embedding for chunks_fixed[0].

        # -----
        if retrieval_type == DENSE:
            # ----- 6. Retrieve
            #retrieve_chunks(retrieval_type, chunk_type, eval_set, chunks, indexing, bm25, k)
            retrieved_output = retrieve_chunks(retrieval_type, chunk_type, eval_set, chunks, indexing, None, k)
            # ----- 7. Evaluate
            eval_metrices = evaluate(retrieval_type, chunk_type, n_samples, retrieved_output,
                            use_faithfulness=False, use_relevancy=False, use_llm_correctness=False)

    if retrieval_type == BM25 or retrieval_type == HYBRID:

        # ----- 4. Bm25 indexing
        bm25 = generate_bm25_indexing(chunks)

        if retrieval_type == BM25:
            # ----- 5. Retrieve
            #retrieve_chunks(retrieval_type, chunk_type, eval_set, chunks, indexing, bm25, k)
            retrieved_output = retrieve_chunks(retrieval_type, chunk_type, eval_set, chunks, None, bm25, k)
            # ----- 6. Evaluate
            eval_metrices = evaluate(retrieval_type, chunk_type, n_samples, retrieved_output,
                            use_faithfulness=False, use_relevancy=False, use_llm_correctness=False)
    
    if retrieval_type == HYBRID:
        # ----- 5. Retrieve
        #retrieve_chunks(retrieval_type, chunk_type, eval_set, chunks, indexing, bm25, k)
        retrieved_output = retrieve_chunks(retrieval_type, chunk_type, eval_set, chunks, indexing, bm25, k)
        # ----- 6. Evaluate
        eval_metrices = evaluate(retrieval_type, chunk_type, n_samples, retrieved_output,
                        use_faithfulness=False, use_relevancy=False, use_llm_correctness=False)
        



