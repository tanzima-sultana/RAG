import os
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
import pickle 
import pyarrow.parquet as pq
import shutil
import boto3
import io

from config import S3_BUCKET
from constants import INDEX_FLATIP, INDEX_IVF, INDEX_HNSW, LOCAL, AWS
from src.dist import s3_utills

class Indexing:
    def __init__(self, mode, dataset_size, device):
        self.mode = mode 
        self.dataset_size = dataset_size
        self.device = device

        self.flatip_path = f"index/flatip/"
        self.ivf_path = f"index/ivf/"
        self.hnsw_path = f"index/hnsw/"

        self.faiss_ids_path = f"index/faiss_ids/"

        self.bm25_path = f"index/bm25/"
        self.bm25_ids_path = f"index/bm25_ids/"

        if mode == AWS:
            self.flatip_path = f"s3://{S3_BUCKET}/" + self.flatip_path
            self.ivf_path = f"s3://{S3_BUCKET}/" + self.ivf_path
            self.hnsw_path = f"s3://{S3_BUCKET}/" + self.hnsw_path

            self.faiss_ids_path = f"s3://{S3_BUCKET}/" + self.faiss_ids_path

            self.bm25_path = f"s3://{S3_BUCKET}/" + self.bm25_path
            self.bm25_ids_path = f"s3://{S3_BUCKET}/" + self.bm25_ids_path


    def is_exists(self, path):
        if self.mode == AWS:
            return s3_utills.s3_file_exists(path)
        else:
            return os.path.exists(path)

    # -------- shared id save/load -------- #

    def save_ids(self, path, chunk_ids):
        if self.mode == LOCAL:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(chunk_ids, f)
        else:
            import s3fs
            fs = s3fs.S3FileSystem()
            with fs.open(path, 'wb') as f:
                pickle.dump(chunk_ids, f)
        print(f"Saved chunk_ids to {path}")

    def load_ids(self, path):
        if self.mode == AWS:
            import s3fs
            fs = s3fs.S3FileSystem()
            with fs.open(path, 'rb') as f:
                return pickle.load(f)
        else:
            with open(path, 'rb') as f:
                return pickle.load(f)

    # -------- FAISS index -------- #
    
    def save_faiss(self, index, path):
        if self.mode == LOCAL:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            faiss.write_index(index, path)
        else:
            import s3fs
            local_tmp = "/tmp/index.faiss"
            faiss.write_index(index, local_tmp)

            fs = s3fs.S3FileSystem()
            fs.put(local_tmp, path)  
        
        print(f"Saved index to {path}")
    
    def load_faiss(self, path):
        print(f"Load indexing FAISS-faltip from disk")
        if self.mode == AWS:
            import s3fs
            local_tmp = "/tmp/index.faiss"
            fs = s3fs.S3FileSystem()
            fs.get(path, local_tmp)
            return faiss.read_index(local_tmp)
        else:
            return faiss.read_index(path)
    
    def generate_flat_ip(self, embeddings, index_path):

        if self.is_exists(index_path):
            return self.load_faiss(index_path)
        
        # Generate index
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        # Save
        self.save_faiss(index, index_path)
    
    def generate_ivf(self, embeddings, index_path, nlist):

        if self.is_exists(index_path):
            return self.load_faiss(index_path)
        
        # Generate index
        dim = embeddings.shape[1]

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        min_train_size = 39 * nlist
        if embeddings.shape[0] < min_train_size:
            print(f"WARNING: {embeddings.shape[0]} vectors < recommended {min_train_size} for nlist={nlist}.")

        index.train(embeddings)
        index.add(embeddings)

        # Save
        self.save_faiss(index, index_path)

    def generate_hnsw(self, embeddings, index_path, M):

        if self.is_exists(index_path):
            return self.load_faiss(index_path)

        # Generate index
        dim = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        index.add(embeddings)

        # Save
        self.save_faiss(index, index_path)
        
    def generate_faiss_index(self, embedding_path, ivf_nlist, hnsw_m):
        path_extn = embedding_path.removeprefix("embeddings/").removesuffix(".pkl")
        
        path1 = self.flatip_path + path_extn
        path2 = self.ivf_path + path_extn
        path3 = self.hnsw_path + path_extn
        ids_path = self.faiss_ids_path + path_extn

        if self.is_exists(path1) and self.is_exists(path2) and self.is_exists(path3) and self.is_exists(ids_path):
            print("Load faiss-indexing from disk")
            return path1, path2, path3, ids_path

        # Load embedding_map
        embeddings_map = None
        with open(embedding_path, "rb") as f:
            embeddings_map = pickle.load(f)
        
        try:
            # Load embedding
            chunk_ids = list(embeddings_map.keys())
            em = list(embeddings_map.values())
            embeddings = np.array(em).astype('float32')

            # Faltip
            self.generate_flat_ip(embeddings, path1)

            # IVF
            self.generate_ivf(embeddings, path2, ivf_nlist)

            # HNSW
            self.generate_hnsw(embeddings, path3, hnsw_m)

            self.save_ids(ids_path, chunk_ids)

        except Exception as e:
            print(f"Indexing failed: {e}")
            # remove dir if fails
            if os.path.exists(path1):
                shutil.rmtree(path1)
            if os.path.exists(path2):
                shutil.rmtree(path2)
            if os.path.exists(path3):
                shutil.rmtree(path3)
            if os.path.exists(ids_path):
                shutil.rmtree(ids_path)

            return None, None, None, None

        return path1, path2, path3, ids_path
        

    # ------------- BM25 ---------- #

    def tokenize_chunk_text(self, text):
        return text.lower().split()

    def save_bm25(self, index, path):
        if self.mode == AWS:
            import s3fs
            fs = s3fs.S3FileSystem()
            with fs.open(path, 'wb') as f:
                pickle.dump(index, f)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(index, f)
        print(f"Saved BM25 index to {path}")

    def generate_bm25_index(self, chunk_path):

        path_extn = chunk_path.removeprefix("chunks/").removesuffix(".pkl")
        path = self.bm25_path + path_extn
        ids_path = self.bm25_ids_path + path_extn

        if self.is_exists(path) and self.is_exists(ids_path):
            print("Load bm25-indexing from disk")
            return path, ids_path

        bm25_index = None
        try:
            # Load chunks
            chunks_map = None
            with open(chunk_path, "rb") as f:
                chunks_map = pickle.load(f)

            chunk_ids = list(chunks_map.keys())
            chunks = list(chunks_map.values())

            # generate bm25 index
            tokens = [self.tokenize_chunk_text(chunk['chunk_text']) for chunk in chunks]
            bm25_index = BM25Okapi(tokens)

            # Save
            self.save_bm25(bm25_index, path)
            self.save_ids(ids_path, chunk_ids)
        
        except Exception as e:
            print(f"Index generation failed: {e}")
            # remove dir if fails
            if os.path.exists(path):
                shutil.rmtree(path)
            if os.path.exists(ids_path):
                shutil.rmtree(ids_path)
            return None, None
        
        return path, ids_path