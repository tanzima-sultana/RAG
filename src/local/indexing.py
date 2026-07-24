import os
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
import pickle 
from constants import INDEX_FLATIP, INDEX_IVF, INDEX_HNSW

class Indexing:
    def __init__(self, dataset_size, device, chunking_type, indexing_type):
        self.dataset_size = dataset_size
        self.device = device
        self.chunking_type = chunking_type 
        self.indexing_type = indexing_type

        self.faiss_path = f"index/idx_{dataset_size}_{device}_{chunking_type}_{indexing_type}"
        self.bm25_path = f"index/bm25_{dataset_size}_{device}_{chunking_type}"

    def save_faiss(self, index):
        os.makedirs(os.path.dirname(self.faiss_path), exist_ok=True)
        faiss.write_index(index, self.faiss_path)
        print(f"Saved index to {self.faiss_path}")
        

    def generate_flat_ip(self, embeddings):
        embeddings = np.array(embeddings).astype('float32')
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        self.save_faiss(index)
        return index

    def generate_ivf(self, embeddings, nlist=256):
        embeddings = np.array(embeddings).astype('float32')
        dim = embeddings.shape[1]

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        min_train_size = 39 * nlist
        if embeddings.shape[0] < min_train_size:
            print(f"WARNING: {embeddings.shape[0]} vectors < recommended {min_train_size} for nlist={nlist}.")

        index.train(embeddings)
        index.add(embeddings)

        self.save_faiss(index)
        return index

    def generate_hnsw(self, embeddings, M=32):
        
        embeddings = np.array(embeddings).astype('float32')
        dim = embeddings.shape[1]

        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        index.add(embeddings)

        self.save_faiss(index)
        return index
    
    def generate_faiss_index(self, embeddings):
        
        if os.path.exists(self.faiss_path):
            print(f"Load indexing {self.indexing_type} from disk")
            return faiss.read_index(self.faiss_path)
        
        if self.indexing_type == INDEX_FLATIP:
            return self.generate_flat_ip(embeddings)
        elif self.indexing_type == INDEX_IVF:
            return self.generate_ivf(embeddings, nlist=256)
        else:
            return self.generate_hnsw(embeddings, M=32)

    # ------------- Bm25 ---------- #

    def tokenize_chunk_text(self, text):
        return text.lower().split()

    def save_bm25(self, index):
        os.makedirs(os.path.dirname(self.bm25_path), exist_ok=True)
        with open(self.bm25_path, 'wb') as f:
            pickle.dump(index, f)
        print(f"Saved BM25 index to {self.bm25_path}")

    def generate_bm25_index(self, chunks):
        if os.path.exists(self.bm25_path):
            print("Load BM25 index from disk")
            with open(self.bm25_path, 'rb') as f:
                return pickle.load(f)

        tokens = [self.tokenize_chunk_text(chunk['chunk_text']) for chunk in chunks]
        bm25_index = BM25Okapi(tokens)
        self.save_bm25(bm25_index)
        return bm25_index