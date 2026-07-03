from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt

import os
import pickle

from dataset import load_processed_dataset
from chunking import compute_chunks, fixed_chunking, sentence_aware_chunking, semantic_chunking
from embedding import generate_embeddings
from faiss_indexing import generate_FlatIndexIP
from eval import evaluate
from generate_answer import get_single_answer

model = SentenceTransformer('all-MiniLM-L6-v2')

# ----- main
if __name__ == "__main__":  

    # 1. Load dataset
    n_samples = 100
    dataset = load_processed_dataset()
    sample = dataset.select(range(n_samples)) 

    # 2. Chunking
    chunks_fixed = compute_chunks(fixed_chunking, sample, n_samples, 'fixed')
    chunks_sentence = compute_chunks(sentence_aware_chunking, sample, n_samples, 'sentence-aware')
    chunks_semantic = compute_chunks(semantic_chunking, sample, n_samples, 'semantic')

    #print(chunks_fixed[0])

    # 3. Embedding

    embeddings_fixed = generate_embeddings(chunks_fixed, n_samples, 'fixed')
    embeddings_sentence = generate_embeddings(chunks_sentence, n_samples, 'sentence-aware')
    embeddings_semantic = generate_embeddings(chunks_semantic, n_samples, 'semantic')

    #print("Fixed embeddings shape: ", np.array(embeddings_fixed).shape)
    #print("Sentence embeddings shape: ", np.array(embeddings_sentence).shape)
    #print("Semantic embeddings shape: ", np.array(embeddings_semantic).shape)

    # 4. FAISS indexing

    indexing_fixed = generate_FlatIndexIP(embeddings_fixed, n_samples, 'fixed')
    indexing_sentence = generate_FlatIndexIP(embeddings_sentence, n_samples, 'sentence-aware')
    indexing_semantic = generate_FlatIndexIP(embeddings_semantic, n_samples, 'semantic')

    # All of these maintain same indexing strategy, so we can use the same index for all of them. indexing_fixed[0] has the indexing for
    # embeddings_fixed[0] and embeddings_fixed[0] has the embedding for chunks_fixed[0].

    # 5. Evaluation
    k = 5
    print("--- k : ", k)
    #eval = evaluate('fixed', sample, n_samples, chunks_fixed, indexing_fixed, k,
    #                use_faithfulness=False, use_relevancy=False, use_llm_correctness=False)
    #print("eval)



    # 6. Generate answer 

    # Using fixed chunks
    #qus = "Who is the author of outlander?"
    #ans = get_single_answer(chunks_fixed, indexing_fixed, qus, k)

    #print("Ans - ", ans)



