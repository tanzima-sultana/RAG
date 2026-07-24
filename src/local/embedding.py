from sentence_transformers import SentenceTransformer
import numpy as np
import os 

class Embedding:
    def __init__(self, model_name, dataset_size, device, chunking_type):
        self.model_name = model_name
        self.dataset_size = dataset_size
        self.device = device
        self.chunking_type = chunking_type

        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.path = f"embeddings/em_{dataset_size}_{device}_{chunking_type}.npy"
    
    def generate_embeddings(self, chunks):
        
        if os.path.exists(self.path):
            with open(self.path, 'rb') as f:
                print("Load embeddings from disk")
                return np.load(f)

        embeddings = self.model.encode([chunk['chunk_text'] for chunk in chunks], normalize_embeddings=True)

        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'wb') as f:
            np.save(f, embeddings)

        return embeddings