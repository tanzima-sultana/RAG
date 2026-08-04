from sentence_transformers import SentenceTransformer
import numpy as np
import os 
import pyarrow as pa
import pyarrow.parquet as pq
import pickle
import shutil
import time 

class Embedding:
    def __init__(self, dataset_size, device, model_name):
        
        self.dataset_size = dataset_size
        self.device = device
        self.model_name = model_name

        self.model = SentenceTransformer(self.model_name, device=self.device)

        self.embedding_path = f"embeddings/"

    def generate_embeddings(self, chunk_path):
        
        out_path = self.embedding_path + chunk_path.removeprefix("chunks/")
        print("\nEmbedding path : ", out_path)

        if os.path.exists(out_path):
            print("Load embeddings from disk")
            return out_path

        try:
            # Load chunks
            chunks_map = None
            with open(chunk_path, "rb") as f:
                chunks_map = pickle.load(f)

            chunk_ids = chunks_map.keys()
            chunks = chunks_map.values()

            # Embed
            s = time.time()
            embeddings = self.model.encode(
                [chunk['chunk_text'] for chunk in chunks], normalize_embeddings=True
            )
            elapsed_time = time.time() - s
            print("time : ", elapsed_time)

            # Map
            embedding_map = dict(zip(chunk_ids, embeddings))

            # Save embeddings
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                pickle.dump(embedding_map, f)
            
            embedding_size = os.path.getsize(out_path) / (1024**2)
            print("Embedding size : ", embedding_size)

        except Exception as e:
            print(f"Embedding failed: {e}")
            # remove dir if fails
            if os.path.exists(out_path):
                shutil.rmtree(out_path)
            return None
        
        return out_path