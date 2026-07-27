import os
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
import pickle 
import pyarrow.parquet as pq

import boto3
import io

from config import S3_BUCKET
from constants import INDEX_FLATIP, INDEX_IVF, INDEX_HNSW, LOCAL, AWS
from src.dist import s3_utills

class Indexing:
    def __init__(self, mode, dataset_size, device, chunking_type, indexing_type):
        self.mode = mode 
        self.dataset_size = dataset_size
        self.device = device
        self.chunking_type = chunking_type 
        self.indexing_type = indexing_type

        self.faiss_path = f"index/idx_{mode}_{dataset_size}_{device}_{chunking_type}_{indexing_type}"
        self.bm25_path = f"index/bm25_{mode}_{dataset_size}_{device}_{chunking_type}"

        if mode == AWS:
            self.faiss_path = f"s3://{S3_BUCKET}/" + self.faiss_path
            self.bm25_path = f"s3://{S3_BUCKET}/" + self.bm25_path


    def is_exists(self, path):
        if self.mode == AWS:
            return s3_utills.s3_file_exists(path)
        else:
            return os.path.exists(path)
    # -------- FAISS index 
    
    def save_faiss(self, index):

        if self.mode == LOCAL:
            os.makedirs(os.path.dirname(self.faiss_path), exist_ok=True)
            faiss.write_index(index, self.faiss_path)
        else:
            import s3fs
            local_tmp = "/tmp/index.faiss"
            faiss.write_index(index, local_tmp)

            fs = s3fs.S3FileSystem()
            fs.put(local_tmp, self.faiss_path)  
        
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
    
    def generate_faiss_index(self, embedding_path):

        if self.is_exists(self.faiss_path):
            print(f"Load indexing {self.indexing_type} from disk")
            if self.mode == AWS:
                import s3fs
                local_tmp = "/tmp/index.faiss"
                fs = s3fs.S3FileSystem()
                fs.get(self.faiss_path, local_tmp)
                return faiss.read_index(local_tmp)
            else:
                return faiss.read_index(self.faiss_path)

        faiss_index = None 
        try:
            table = pq.read_table(embedding_path)
            embeddings = np.array(table["embedding"].to_pylist(), dtype=np.float32)

            if self.indexing_type == INDEX_FLATIP:
                faiss_index = self.generate_flat_ip(embeddings)
            elif self.indexing_type == INDEX_IVF:
                faiss_index = self.generate_ivf(embeddings, nlist=256)
            else:
                faiss_index = self.generate_hnsw(embeddings, M=32)
        
        except Exception as e:
            print(f"Index generation failed: {e}")
            return None
        
        if not self.is_exists(self.faiss_path):
            print("Index write produced no output")
            return None
        
        return faiss_index

    # ------------- Bm25 ---------- #

    def tokenize_chunk_text(self, text):
        return text.lower().split()

    def save_bm25(self, index):
        if self.mode == AWS:
            import s3fs
            fs = s3fs.S3FileSystem()
            with fs.open(self.bm25_path, 'wb') as f:
                pickle.dump(index, f)
        else:
            os.makedirs(os.path.dirname(self.bm25_path), exist_ok=True)
            with open(self.bm25_path, 'wb') as f:
                pickle.dump(index, f)
        print(f"Saved BM25 index to {self.bm25_path}")

    def generate_bm25_index(self, chunk_path):
        if self.is_exists(self.bm25_path):
            if self.mode == AWS:
                import s3fs
                fs = s3fs.S3FileSystem()
                with fs.open(self.bm25_path, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(self.bm25_path, 'rb') as f:
                    return pickle.load(f)

        bm25_index = None
        try:
            table = pq.read_table(chunk_path)
            chunks = table.to_pylist()

            tokens = [self.tokenize_chunk_text(chunk['chunk_text']) for chunk in chunks]
            bm25_index = BM25Okapi(tokens)
            self.save_bm25(bm25_index)
        
        except Exception as e:
            print(f"Index generation failed: {e}")
            return None
        
        if not self.is_exists(self.bm25_path):
            print("Index write produced no output")
            return None
        
        return bm25_index