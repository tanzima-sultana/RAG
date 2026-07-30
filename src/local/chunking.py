
import os
import pickle
import numpy as np
import pandas as pd
import shutil
import time 
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

from constants import FIXED, SENTENCE, SEMANTIC

class Chunking:
    def __init__(self, dataset, dataset_size, device, model_name):
        self.model_name = model_name
        self.dataset = dataset
        self.dataset_size = dataset_size
        self.device = device

        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.tokenizer = self.model.tokenizer

        self.fixed_path = f"chunks/{dataset_size}_{FIXED}/"
        self.sentence_path = f"chunks/{dataset_size}_{SENTENCE}/"
        self.semantic_path = f"chunks/{dataset_size}_{SEMANTIC}/"
    

    def CHUNK(self, doc_id, title, chunk_id, chunk_text, chunk_size, chunking_type):
        return {
            'doc_id': doc_id,
            'title': title,
            'chunk_id': f"{doc_id}_{chunk_id}",
            'chunk_text': chunk_text,
            'chunk_size' : chunk_size,
            'chunking_type': chunking_type
        }

    def _is_exists(self):
        return os.path.exists(self.fixed_path) and os.path.exists(self.sentence_path) and os.path.exists(self.semantic_path)
            
    def load(self):
        with open(self.fixed_path, "rb") as f:
            fixed_chunks = pickle.load(f)
        with open(self.sentence_path, "rb") as f:
            sentence_chunks = pickle.load(f)
        with open(self.semantic_path, "rb") as f:
            semantic_chunks = pickle.load(f)
        
        return fixed_chunks, sentence_chunks, semantic_chunks

    def save(self, chunks, out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, 'wb') as f:
            pickle.dump(chunks, f)

    # -------------- 1. Fixed
    def fixed_chunking(self, doc_id, title, text, max_chunk_size, fix_chunk_overlap):
        #print("Fixed chunking")
        s = time.time()

        chunks = {}
        chunk_sizes = []

        
        # Get the tokens from the text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        start = 0
        chunk_id = 0

        while start < len(tokens):
            end = start + max_chunk_size
            # Get the tokens between start and end, chunk boundary
            chunk_tokens = tokens[start:end]
            # Get text from tokens. Decode
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            
            # Make CHUNK
            ch = self.CHUNK(doc_id, title, chunk_id, chunk_text, len(chunk_tokens), FIXED)
            chunks[ch['chunk_id']] = ch 

            chunk_sizes.append(len(chunk_tokens))
            
            # To ensure overlap
            start += max_chunk_size - fix_chunk_overlap
            chunk_id += 1

        elapsed_time = time.time() - s 
        #print(elapsed_time)
        return chunks, chunk_sizes, elapsed_time

    # ----- 2. Sentence aware chunking
    def sentence_aware_chunking(self, doc_id, title, text, max_chunk_size, fix_chunk_overlap):  
        s = time.time()

        chunks = {}
        chunk_sizes = [] 

         # Get the sentences
        sentences = sent_tokenize(text)
        
        current_chunk = []
        current_chunk_tokens = 0
        chunk_id = 0

        for sentence in sentences:
            # Get token for each sentence, count tokens
            tokens = self.tokenizer.encode(sentence, add_special_tokens=False)

            # 1. if a sentence token exceeds the max_chunk_size, we will have to split it into multiple chunks using fixed chunking
            # not adding it made long sentences skip totally
            if len(tokens) > max_chunk_size:
                # If some chunks already there need to add them
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    ch = self.CHUNK(doc_id, title, chunk_id, chunk_text, current_chunk_tokens, SENTENCE)
                    chunks[ch['chunk_id']] = ch 
                    chunk_sizes.append(current_chunk_tokens)
                    chunk_id += 1

                start = 0
                while start < len(tokens):
                    end = start + max_chunk_size
                    # Get the tokens between start and end, chunk boundary
                    chunk_tokens = tokens[start:end]
                    # Get text from tokens. Decode
                    chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                    
                    # Make CHUNK
                    ch = self.CHUNK(doc_id, title, chunk_id, chunk_text, len(chunk_tokens), SENTENCE)
                    chunks[ch['chunk_id']] = ch 
                    chunk_sizes.append(len(chunk_tokens))

                    # To ensure overlap
                    start += max_chunk_size - fix_chunk_overlap
                    chunk_id += 1

                # Reset for next chunk
                current_chunk = []
                current_chunk_tokens = 0

                # Move to the next sentence after splitting the long sentence into chunks
                continue  

            # 2. If adding the current sentence exceeds the max_chunk_size, finalize the current chunk and start a new one
            elif current_chunk_tokens + len(tokens) > max_chunk_size:
                chunk_text = ' '.join(current_chunk)

                # Make CHUNK
                ch = self.CHUNK(doc_id, title, chunk_id, chunk_text, current_chunk_tokens, SENTENCE)
                chunks[ch['chunk_id']] = ch 
                chunk_sizes.append(current_chunk_tokens)

                # Reset for next chunk
                current_chunk = []
                current_chunk_tokens = 0
                chunk_id += 1
            
            # 3. if above two condition does not meet, proceed normally
            # Append the sentence/the words from the sentences to current chunk
            current_chunk.append(sentence)
            current_chunk_tokens += len(tokens)
        
        # ---------- Loop end 

        # Add the last chunk if it has content
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            # Make CHUNK
            ch = self.CHUNK(doc_id, title, chunk_id, chunk_text, current_chunk_tokens, SENTENCE)
            chunks[ch['chunk_id']] = ch 
            chunk_sizes.append(current_chunk_tokens)

        # Return
        elapsed_time = time.time() - s 
        return chunks, chunk_sizes, elapsed_time


    # --------------- 3. Semantic chunking
    def semantic_chunking(self, doc_id, title, text, max_chunk_size, fix_chunk_overlap, semantic_threshold):
        s = time.time()

        chunks = {}
        chunk_sizes = []

        sentences = sent_tokenize(text)

        # Precompute all sentence embeddings in one batched call
        sentence_embeddings = self.model.encode(sentences, batch_size=64, show_progress_bar=False)

        current_chunk = []
        current_chunk_tokens = 0
        chunk_id = 0
        prev_embedding = None

        for i in range(len(sentences)):
            sentence = sentences[i]
            tokens = self.tokenizer.encode(sentence, add_special_tokens=False)

            # 1. if a sentence token exceeds the max_chunk_size, we will have to split it into multiple chunks using fixed chunking
            # not adding it made long sentences skip totally
            if len(tokens) > max_chunk_size:
                # If some chunks already there need to add them
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    ch = self.CHUNK(doc_id, title, chunk_id, chunk_text, current_chunk_tokens, SEMANTIC)
                    chunks[ch['chunk_id']] = ch 
                    chunk_sizes.append(current_chunk_tokens)
                    chunk_id += 1

                start = 0
                while start < len(tokens):
                    end = start + max_chunk_size
                    chunk_tokens = tokens[start:end]
                    chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)

                    ch = self.CHUNK(doc_id, title, chunk_id, chunk_text, len(chunk_tokens), SEMANTIC)
                    chunks[ch['chunk_id']] = ch 
                    chunk_sizes.append(len(chunk_tokens))

                    start += max_chunk_size - fix_chunk_overlap
                    chunk_id += 1

                prev_embedding = None
                current_chunk = []
                current_chunk_tokens = 0
                continue

            if prev_embedding is not None:
                embedding_1 = prev_embedding
                embedding_2 = sentence_embeddings[i]

                cosine_similarity = np.dot(embedding_1, embedding_2) / (
                    np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2)
                )

                if (cosine_similarity < semantic_threshold) or (current_chunk_tokens + len(tokens) > max_chunk_size):
                    chunk_text = ' '.join(current_chunk)
                    ch = self.CHUNK(doc_id, title, chunk_id, chunk_text, current_chunk_tokens, SEMANTIC)
                    chunks[ch['chunk_id']] = ch 
                    chunk_sizes.append(current_chunk_tokens)
                    current_chunk = []
                    current_chunk_tokens = 0
                    chunk_id += 1

                prev_embedding = embedding_2
            else:
                prev_embedding = sentence_embeddings[i]

            current_chunk.append(sentence)
            current_chunk_tokens += len(tokens)

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            ch = self.CHUNK(doc_id, title, chunk_id, chunk_text, current_chunk_tokens, SEMANTIC)
            chunks[ch['chunk_id']] = ch 
            chunk_sizes.append(current_chunk_tokens)

        elapsed_time = time.time() - s 
        return chunks, chunk_sizes, elapsed_time
            
    # -----------------------
    def compute_chunks(self, max_chunk_size, fix_chunk_overlap, semantic_threshold):

        self.fixed_path = self.fixed_path + f"size_{max_chunk_size}_overlap_{fix_chunk_overlap}.pkl"
        self.sentence_path = self.sentence_path + f"size_{max_chunk_size}_overlap_{fix_chunk_overlap}.pkl"
        self.semantic_path = self.semantic_path + f"size_{max_chunk_size}_overlap_{fix_chunk_overlap}_threashold_{semantic_threshold}.pkl"

        #print("Compute chunks start")
        
        # If alraedy exists, return that
        if self._is_exists():
            print("Read chunks from disk")
            #return self.load()
            return self.fixed_path, self.sentence_path, self.semantic_path
        
        try:
            fixed_chunks = {}
            fixed_chunks_sizes = []
            time1 = 0

            sentence_chunks = {}
            sentence_chunk_sizes = []
            time2 = 0

            semantic_chunks = {}
            semantic_chunk_sizes = []
            time3 = 0

            for i, doc in enumerate(self.dataset):
                #print(f"Processing doc {i}/{len(self.dataset)}")
                doc_id = doc['doc_id']
                title = doc['title']
                text = doc['text']

                c11, c12, t1 = self.fixed_chunking(doc_id, title, text, max_chunk_size, fix_chunk_overlap)
                fixed_chunks.update(c11)
                fixed_chunks_sizes.extend(c12)
                time1 += t1

                c21, c22, t2 = self.sentence_aware_chunking(doc_id, title, text, max_chunk_size, fix_chunk_overlap)
                sentence_chunks.update(c21) 
                sentence_chunk_sizes.extend(c22)
                time2 += t2

                c31, c32, t3 = self.semantic_chunking(doc_id, title, text, max_chunk_size, fix_chunk_overlap, semantic_threshold)
                semantic_chunks.update(c31)
                semantic_chunk_sizes.extend(c32)
                time3 += t3
            
            # Save
            self.save(fixed_chunks, self.fixed_path)
            self.save(sentence_chunks, self.sentence_path)
            self.save(semantic_chunks, self.semantic_path)

            print("Avg chunk sizes : fixed, sentence, semantic : ", np.mean(fixed_chunks_sizes), np.mean(sentence_chunk_sizes), np.mean(semantic_chunk_sizes))
            print("Total time for chunks : fixed, sentence, semantic : ", time1, time2, time3)

            return self.fixed_path, self.sentence_path, self.semantic_path

        except Exception as e:
            print(f"Chunking failed: {e}")
            # remove dir if fails
            if os.path.exists(self.fixed_path):
                shutil.rmtree(self.fixed_path)
            if os.path.exists(self.sentence_path):
                shutil.rmtree(self.sentence_path)
            if os.path.exists(self.semantic_path):
                shutil.rmtree(self.semantic_path)

            return None, None, None