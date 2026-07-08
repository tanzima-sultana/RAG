
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle
import faiss

def generate_FlatIndexIP(embeddings, dataset_size, strategy_name):
    path = f"indexing_FlatIndexIP/{dataset_size}/{strategy_name}.pkl"

    #print(strategy_name, path)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            #print("Load indexing (FlatIndexIP) from disk")
            return pickle.load(f)

    # Convert embeddings to float32
    embeddings = np.array(embeddings).astype('float32')

    # Create a FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])  

    # Add embeddings to the index
    # Its an array of vector, so fixed_embedding is 342X384, it has 342 vectors with 384 diemnsion saved sequentially.
    # It will return an index for search query. The closest vector index of the query vector.
    index.add(embeddings)

    # Use pickle to save index because index in faiss object. not numpy array 
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(index, f)

    return index 
