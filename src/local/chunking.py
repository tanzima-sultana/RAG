
import os
import pickle
import numpy as np

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

from constants import FIXED, SENTENCE, SEMANTIC

class Chunking:
    def __init__(self, model_name, dataset, dataset_size, device, chunking_type):
        self.model_name = model_name
        self.dataset = dataset
        self.dataset_size = dataset_size
        self.device = device
        self.chunking_type = chunking_type

        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.tokenizer = self.model.tokenizer

        self.path = f"chunks/ch_{dataset_size}_{device}_{chunking_type}.pkl"
    

    def CHUNK(self, doc_id, title, chunk_id, chunk_text, chunk_size, chunking_type):
        return {
            'doc_id': doc_id,
            'title': title,
            'chunk_id': f"{doc_id}_{chunk_id}",
            'chunk_text': chunk_text,
            'chunk_size' : chunk_size,
            'chunking_type': chunking_type
        }

    # -------------- 1. Fixed
    def fixed_chunking(self, doc_id, title, text, max_chunk_size, fix_chunk_overlap):
        chunks = []
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
            chunks.append(self.CHUNK(doc_id, title, chunk_id, chunk_text, len(chunk_tokens), self.chunking_type))
            chunk_sizes.append(len(chunk_tokens))
            
            # To ensure overlap
            start += max_chunk_size - fix_chunk_overlap
            chunk_id += 1

        
        return chunks, chunk_sizes

    # ----- 2. Sentence aware chunking
    def sentence_aware_chunking(self, doc_id, title, text, max_chunk_size, fix_chunk_overlap):  
        chunks = []
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
                    chunks.append(self.CHUNK(doc_id, title, chunk_id, chunk_text, current_chunk_tokens, self.chunking_type))
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
                    chunks.append(self.CHUNK(doc_id, title, chunk_id, chunk_text, len(chunk_tokens), self.chunking_type))
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
                chunks.append(self.CHUNK(doc_id, title, chunk_id, chunk_text, current_chunk_tokens, self.chunking_type))
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
            chunks.append(self.CHUNK(doc_id, title, chunk_id, chunk_text, current_chunk_tokens, self.chunking_type))
            chunk_sizes.append(current_chunk_tokens)

        # Return
        return chunks, chunk_sizes


    # --------------- 3. Semantic chunking
    def semantic_chunking(self, doc_id, title, text, max_chunk_size, fix_chunk_overlap, semantic_threshold):
        chunks = []
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
                    chunks.append(self.CHUNK(doc_id, title, chunk_id, chunk_text, current_chunk_tokens, self.chunking_type))
                    chunk_sizes.append(current_chunk_tokens)
                    chunk_id += 1

                start = 0
                while start < len(tokens):
                    end = start + max_chunk_size
                    chunk_tokens = tokens[start:end]
                    chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)

                    chunks.append(self.CHUNK(doc_id, title, chunk_id, chunk_text, len(chunk_tokens), self.chunking_type))
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
                    chunks.append(self.CHUNK(doc_id, title, chunk_id, chunk_text, current_chunk_tokens, self.chunking_type))
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
            chunks.append(self.CHUNK(doc_id, title, chunk_id, chunk_text, current_chunk_tokens, self.chunking_type))
            chunk_sizes.append(current_chunk_tokens)

        return chunks, chunk_sizes
            
    # -----------------------
    def compute_chunks(self, max_chunk_size, fix_chunk_overlap, semantic_threshold):

        #print("Compute chunks start")
        
        # If alraedy exists, return that
        if os.path.exists(self.path):
            print("Read chunks from disk : ", self.chunking_type)
            with open(self.path, 'rb') as f:
                cached = pickle.load(f)
            return self.path, cached['chunks'], cached['avg_chunk_size']
        
        try:
            chunks = []
            chunks_sizes = []

            for i, doc in enumerate(self.dataset):
                #print(f"Processing doc {i}/{len(self.dataset)}")
                doc_id = doc['doc_id']
                title = doc['title']
                text = doc['text']
                
                c1 = []
                c2 = []
                if self.chunking_type == FIXED:
                    c1, c2 = self.fixed_chunking(doc_id, title, text, max_chunk_size, fix_chunk_overlap)
                elif self.chunking_type == SENTENCE:
                    c1, c2 = self.sentence_aware_chunking(doc_id, title, text, max_chunk_size, fix_chunk_overlap)
                else:
                    c1, c2 = self.semantic_chunking(doc_id, title, text, max_chunk_size, fix_chunk_overlap, semantic_threshold)
                
                chunks.extend(c1)
                chunks_sizes.extend(c2)
        
            avg_chunk_size = np.mean(chunks_sizes)
            
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, 'wb') as f:
                pickle.dump({'chunks': chunks, 'avg_chunk_size': avg_chunk_size}, f)
            
            return self.path, chunks, avg_chunk_size
        
        except Exception as e:
            print(f"Chunking failed: {e}")
            return None, None, None