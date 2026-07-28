from sentence_transformers import SentenceTransformer
import numpy as np
import os 
import pyarrow as pa
import pyarrow.parquet as pq

class Embedding:
    def __init__(self, model_name, dataset_size, device, chunking_type):
        self.model_name = model_name
        self.dataset_size = dataset_size
        self.device = device
        self.chunking_type = chunking_type

        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.path = f"embeddings/em_{dataset_size}_{device}_{chunking_type}.parquet"

    def generate_embeddings(self, chunks):
        if os.path.exists(self.path):
            print("Load embeddings from disk")
            return self.path

        try:
            embeddings = self.model.encode(
                [chunk['chunk_text'] for chunk in chunks], normalize_embeddings=True
            )

            table = pa.table({
                "chunk_id": [chunk['chunk_id'] for chunk in chunks],
                "embedding": [emb.tolist() for emb in embeddings],
            })

            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            pq.write_table(table, self.path)

            return self.path

        except Exception as e:
            print(f"Embedding failed: {e}")
            return None