from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
import os 

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(chunks, dataset_size, strategy_name):
    path = f"embeddings/{dataset_size}/{strategy_name}.npy"
    #print(strategy_name, path)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            #print("Load embeddings from disk" + strategy_name + path)
            return np.load(f)

    embeddings = model.encode([chunk['chunk_text'] for chunk in chunks], normalize_embeddings=True)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        np.save(f, embeddings)
    
    #print("Compute embeddings" + strategy_name + path)

    return embeddings