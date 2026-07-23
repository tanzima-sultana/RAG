
import os
import pickle
import numpy as np

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

from constants import FIXED, SENTENCE, SEMANTIC

class Chunking:
    def __init__(self, dataset, dataset_size, mode, device, chunking_type):
        self.dataset = dataset
        self.dataset_size = dataset_size
        self.mode = mode
        self.device = device
        self.chunking_type = chunking_type

        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = self.model.tokenizer

        self.path = f"chunks/{dataset_size}_{mode}_{device}_{chunking_type}.pkl"
    

    def CHUNK(self, doc_id, title, chunking_type, chunk_id, chunk_text, chunk_size):
        return {
            'doc_id': doc_id,
            'title': title,
            'chunking_type': chunking_type,
            'chunk_id': f"{doc_id}_{chunk_id}",
            'chunk_text': chunk_text,
            'chunk_size' : chunk_size,
        }

    # -------------- 1. Fixed
    def fixed_chunking(self, doc_id, title, text, max_chunk_size, fix_chunk_overlap):
        chunks = []
        chunk_sizes = []

        # Get the tokens from the text
        tokens = self.tokenizer.encode(text)

        start = 0
        chunk_id = 0

        while start < len(tokens):
            end = start + max_chunk_size
            # Get the tokens between start and end, chunk boundary
            chunk_tokens = tokens[start:end]
            # Get text from tokens. Decode
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            
            # Make CHUNK
            chunks.append(self.CHUNK(doc_id, title, chunk_id, chunk_text, len(chunk_tokens), self.chunking_type))
            chunk_sizes.append(len(chunk_tokens))
            
            # To ensure overlap
            start += max_chunk_size - fix_chunk_overlap
            chunk_id += 1

        
        return chunks, chunk_sizes

    # -----------------------
    def compute_chunks(self, max_chunk_size, fix_chunk_overlap, semantic_threshold):
        
        # If alraedy exists, return that
        if os.path.exists(self.path):
            with open(self.path, 'rb') as f:
                cached = pickle.load(f)
            return cached['chunks'], cached['avg_chunk_size']
        
        chunks = []
        chunks_sizes = []

        for doc in self.dataset:
            doc_id = doc['doc_id']
            title = doc['title']
            text = doc['text']
            
            c1, c2 = None 
            if self.chunking_type == FIXED:
                c1, c2 = self.fixed_chunking(doc_id, title, text, max_chunk_size, fix_chunk_overlap)
            #elif self.chunking_type == SENTENCE:
                #c1, c2 = self.sentence_aware_chunking(doc_id, doc_title, text, max_chunk_size, fix_chunk_overlap)
            #else:
                #c1, c2 = self.semantic_chunking(doc_id, doc_title, text, max_chunk_size, fix_chunk_overlap, semantic_threshold)
            
            chunks.extend(c1)
            chunks_sizes.extend(c2)
    
        avg_chunk_size = np.mean(chunks_sizes)
        
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'wb') as f:
            pickle.dump({'chunks': chunks, 'avg_chunk_size': avg_chunk_size}, f)
        
        return chunks, avg_chunk_size