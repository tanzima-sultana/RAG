import os
import pickle
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pyspark import TaskContext
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, MapType, ArrayType, FloatType

from config import S3_BUCKET
from constants import LOCAL, AWS, FIXED, SENTENCE, SEMANTIC
from src.dist import s3_utills

class Chunking_Embedding:
    def __init__(self, mode, num_partition, model_name, dataset_path, dataset_size, device, chunking_type, em_batch_size):
        
        self.mode = mode
        self.num_partition = num_partition
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.dataset_size = dataset_size
        self.device = device
        self.chunking_type = chunking_type
        self.em_batch_size = em_batch_size

        self.model_name = model_name

        self.chunk_path = f"chunks/{dataset_size}_{chunking_type}.pkl"
        self.embedding_path = f"embeddings/{dataset_size}_{chunking_type}.pkl"

        if self.mode == AWS:
            self.chunk_path = f"s3://{S3_BUCKET}/" + self.chunk_path
            self.embedding_path = f"s3://{S3_BUCKET}/" + self.embedding_path
            

    def is_chunk_exists(self, chunk_path):
        if self.mode == AWS:
            return s3_utills.s3_file_exists(chunk_path)
        else:
            return os.path.exists(chunk_path)
    
    def is_embedding_exists(self, embedding_path):
        if self.mode == AWS:
            return s3_utills.s3_file_exists(embedding_path)
        else:
            return os.path.exists(embedding_path)

        
    @staticmethod
    def CHUNK(doc_id, title, chunk_id, chunk_text, chunk_size, chunking_type):
        return {
            'doc_id': doc_id,
            'title': title,
            'chunk_id': f"{doc_id}_{chunk_id}",
            'chunk_text': chunk_text,
            'chunk_size' : chunk_size,
            'chunking_type': chunking_type
        }

    # 1. Fixed
    @staticmethod
    def fixed_chunking(tokenizer, doc_id, title, text, max_chunk_size, fix_chunk_overlap):

        chunks_map = {}

        # Get the tokens from the text
        tokens = tokenizer.encode(text, add_special_tokens=False)

        start = 0
        chunk_id = 0

        while start < len(tokens):
            end = start + max_chunk_size
            # Get the tokens between start and end, chunk boundary
            chunk_tokens = tokens[start:end]
            # Get text from tokens. Decode
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            # Make CHUNK
            ch = Chunking_Embedding.CHUNK(doc_id, title, chunk_id, chunk_text, len(chunk_tokens), FIXED)
            chunks_map[ch['chunk_id']] = ch
            
            # To ensure overlap
            start += max_chunk_size - fix_chunk_overlap
            chunk_id += 1
        
        return chunks_map

    # 2. Sentence
    @staticmethod
    def sentence_aware_chunking(tokenizer, doc_id, title, text, max_chunk_size, fix_chunk_overlap):
        chunks_map = {}

         # Get the sentences
        sentences = sent_tokenize(text)
        
        current_chunk = []
        current_chunk_tokens = 0
        chunk_id = 0

        for sentence in sentences:
            # Get token for each sentence, count tokens
            tokens = tokenizer.encode(sentence, add_special_tokens=False)

            # 1. if a sentence token exceeds the max_chunk_size, we will have to split it into multiple chunks using fixed chunking
            # not adding it made long sentences skip totally
            if len(tokens) > max_chunk_size:
                # If some chunks already there need to add them
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    ch = Chunking_Embedding.CHUNK(doc_id, title, chunk_id, chunk_text, current_chunk_tokens, SENTENCE)
                    chunks_map[ch['chunk_id']] = ch
                    chunk_id += 1

                start = 0
                while start < len(tokens):
                    end = start + max_chunk_size
                    # Get the tokens between start and end, chunk boundary
                    chunk_tokens = tokens[start:end]
                    # Get text from tokens. Decode
                    chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                    
                    # Make CHUNK
                    ch = Chunking_Embedding.CHUNK(doc_id, title, chunk_id, chunk_text, len(chunk_tokens), SENTENCE)
                    chunks_map[ch['chunk_id']] = ch

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
                ch = Chunking_Embedding.CHUNK(doc_id, title, chunk_id, chunk_text, current_chunk_tokens, SENTENCE)
                chunks_map[ch['chunk_id']] = ch

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
            ch = Chunking_Embedding.CHUNK(doc_id, title, chunk_id, chunk_text, current_chunk_tokens, SENTENCE)
            chunks_map[ch['chunk_id']] = ch

        # Return
        return chunks_map

    # 3. Semantic
    @staticmethod
    def semantic_chunking(model, tokenizer, doc_id, title, text, max_chunk_size, fix_chunk_overlap, semantic_threshold):
        chunks_map = {}

        sentences = sent_tokenize(text)

        # Precompute all sentence embeddings in one batched call
        sentence_embeddings = model.encode(sentences, batch_size=64, show_progress_bar=False)

        current_chunk = []
        current_chunk_tokens = 0
        chunk_id = 0
        prev_embedding = None

        for i in range(len(sentences)):
            sentence = sentences[i]
            tokens = tokenizer.encode(sentence, add_special_tokens=False)

            # 1. if a sentence token exceeds the max_chunk_size, we will have to split it into multiple chunks using fixed chunking
            # not adding it made long sentences skip totally
            if len(tokens) > max_chunk_size:
                # If some chunks already there need to add them
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    ch = Chunking_Embedding.CHUNK(doc_id, title, chunk_id, chunk_text, current_chunk_tokens, SEMANTIC)
                    chunks_map[ch['chunk_id']] = ch
                    chunk_id += 1

                start = 0
                while start < len(tokens):
                    end = start + max_chunk_size
                    chunk_tokens = tokens[start:end]
                    chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

                    ch = Chunking_Embedding.CHUNK(doc_id, title, chunk_id, chunk_text, len(chunk_tokens), SEMANTIC)
                    chunks_map[ch['chunk_id']] = ch

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
                    ch = Chunking_Embedding.CHUNK(doc_id, title, chunk_id, chunk_text, current_chunk_tokens, SEMANTIC)
                    chunks_map[ch['chunk_id']] = ch

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
            ch = Chunking_Embedding.CHUNK(doc_id, title, chunk_id, chunk_text, current_chunk_tokens, SEMANTIC)
            chunks_map[ch['chunk_id']] = ch

        return chunks_map


    # --------------------- Chunks & Embedding --------------- #

    def compute_chunks_embeddings(self, max_chunk_size, fix_chunk_overlap, semantic_threshold):

        if self.is_chunk_exists(self.chunk_path) and self.is_embedding_exists(self.embedding_path):
            print("loading chunks and embedding from disk : ", self.chunk_path, self.embedding_path)
            return self.chunk_path, self.embedding_path

        try:
            # spark
            builder = SparkSession.builder.appName("spark_chunking")
            if self.mode == LOCAL:
                builder = builder.master("local[*]")
            spark = builder.getOrCreate()
            
            # raed parquet data using spark
            df = spark.read.parquet(self.dataset_path)
            df = df.repartition(self.num_partition)

            # ------- Each partition -----------------
            def process_partition(rows, chunk_path, embedding_path, 
                                  model_name, device, chunking_type, 
                                  max_chunk_size, fix_chunk_overlap, semantic_threshold, em_batch_size):

                partition_id = TaskContext.get().partitionId()

                model = SentenceTransformer(model_name, device=device)

                # ---------- Chunking
                tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{model_name}")

                chunk_records = {}
                for row in rows:
                    doc_id = row['doc_id']
                    title = row['title']
                    text = row['text']

                    if chunking_type == FIXED:
                        chunks = Chunking_Embedding.fixed_chunking(tokenizer, doc_id, title, text, max_chunk_size, fix_chunk_overlap)
                    elif chunking_type == SENTENCE:
                        chunks = Chunking_Embedding.sentence_aware_chunking(tokenizer, doc_id, title, text, max_chunk_size, fix_chunk_overlap)
                    else:
                        chunks = Chunking_Embedding.semantic_chunking(model, tokenizer, doc_id, title, text, max_chunk_size, fix_chunk_overlap, semantic_threshold)

                    chunk_records.update(chunks)
                
                if chunk_records:
                    # create dir
                    out_chunk_path = f"{chunk_path}/part-{partition_id:05d}.parquet"
                    os.makedirs(os.path.dirname(out_chunk_path), exist_ok=True)

                    with open(out_chunk_path, 'wb') as f:
                        pickle.dump(chunk_records, f)
                else:
                    yield 0
                    return 
                
                # --------------- Embedding 
                chunk_ids = list(chunk_records.keys())
                chunks = list(chunk_records.values())

                #chunk_ids = [r["chunk_id"] for r in chunks]
                #doc_ids = [r["doc_id"] for r in chunks]
                #titles = [r["title"] for r in chunks]
                texts = [r["chunk_text"] for r in chunks]

                embeddings = model.encode(texts, batch_size=em_batch_size, convert_to_numpy=True)

                embedding_records = {}
                for chunk_id in chunk_ids:
                    embedding_records[chunk_id] = embeddings
                
                if embedding_records:
                    # create dir
                    out_embedding_path = f"{embedding_path}/part-{partition_id:05d}.parquet"
                    os.makedirs(os.path.dirname(out_embedding_path), exist_ok=True)

                    with open(out_embedding_path, 'wb') as f:
                        pickle.dump(embedding_records, f)
                else:
                    yield 0
                    return 

                yield partition_id

            # ----------------------

            partition_ids = df.rdd.mapPartitions(
                lambda rows: process_partition(
                    rows, self.chunk_path, self.embedding_path, self.model_name, self.device, self.chunking_type,
                    max_chunk_size, fix_chunk_overlap, semantic_threshold, self.em_batch_size
                )
            ).collect()    
            spark.stop()

            print(partition_ids)

        except Exception as e:
            print(f"Chunking failed: {e}")
            return None, None
        
        if not self.is_chunk_exists(self.chunk_path) or not self.is_embedding_exists(self.embedding_path):
            print("Chunking or embedding write produced no output")
            return None, None

        return self.chunk_path, self.embedding_path