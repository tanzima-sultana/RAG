
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt

from dataset import load_processed_dataset

dataset = load_processed_dataset()
sample = dataset.select(range(100)) 

def CHUNK(doc_id, doc_title, chunk_id, chunk_text, chunk_size, strategy):
    return {
        'doc_id': doc_id,
        'doc_title': doc_title,
        'chunk_id': f"{doc_id}_{chunk_id}",
        'chunk_text': chunk_text,
        'chunk_size' : chunk_size,
        'strategy': strategy
    }

model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = model.tokenizer
max_chunk_size = 256

# ----- a. Fixed chunking

overlap = 32
chunk_size_fixed = []

def chunking_fixed(doc_id, doc_title, text, max_chunk_size, overlap):
    # Get the tokens from the text
    tokens = tokenizer.encode(text)

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(tokens):
        end = start + max_chunk_size
        # Get the tokens between start and end, chunk boundary
        chunk_tokens = tokens[start:end]
        # Get text from tokens. Decode
        chunk_text = tokenizer.decode(chunk_tokens)
        # Make CHUNK
        chunk_size =  len(chunk_tokens)
        chunk_size_fixed.append(chunk_size)
        chunks.append(CHUNK(doc_id, doc_title, chunk_id, chunk_text, chunk_size, 'fixed'))
        # To ensure overlap
        start += max_chunk_size - overlap
        chunk_id += 1

    
    return chunks

def fixed_chunking(dataset):
    chunks_fixed = []
    for doc in dataset:
        doc_id = doc['doc_id']
        doc_title = doc['title']
        text = doc['text']

        chunks_fixed.extend(chunking_fixed(doc_id, doc_title, text, max_chunk_size, overlap))
    
    return chunks_fixed

# ----- b. Sentence aware chunking
chunk_size_sentence = []

def chunking_sentence_aware(doc_id, doc_title, text, max_chunk_size):  

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0
    chunk_id = 0

    for sentence in sentences:
        # Get token for each sentence, count tokens
        tokens = tokenizer.encode(sentence)

        # if a sentence token exceeds the max_chunk_size, we will have to split it into multiple chunks using fixed chunking
        if len(tokens) > max_chunk_size:
            start = 0
            while start < len(tokens):
                end = start + max_chunk_size
                # Get the tokens between start and end, chunk boundary
                chunk_tokens = tokens[start:end]
                # Get text from tokens. Decode
                chunk_text = tokenizer.decode(chunk_tokens)
                # Make CHUNK
                chunk_size =  len(chunk_tokens)
                chunk_size_sentence.append(chunk_size)
                chunks.append(CHUNK(doc_id, doc_title, chunk_id, chunk_text, chunk_size, 'sentence-aware'))
                # To ensure overlap
                start += max_chunk_size - overlap
                chunk_id += 1
            current_chunk = []
            current_chunk_tokens = 0
            continue  # Move to the next sentence after splitting the long sentence into chunks

        # If adding the current sentence exceeds the max_chunk_size, finalize the current chunk and start a new one
        elif current_chunk_tokens + len(tokens) > max_chunk_size:
            chunk_text = ' '.join(current_chunk)
            chunk_size = current_chunk_tokens
            chunk_size_sentence.append(chunk_size)
            chunks.append(CHUNK(doc_id, doc_title, chunk_id, chunk_text, chunk_size, 'sentence-aware'))

            current_chunk = []
            current_chunk_tokens = 0
            chunk_id += 1
        
        # Append the sentence/the words from the sentences to current chunk
        current_chunk.append(sentence)
        current_chunk_tokens += len(tokens)
    
    # Add the last chunk if it has content
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunk_size = current_chunk_tokens
        chunk_size_sentence.append(chunk_size)
        chunks.append(CHUNK(doc_id, doc_title, chunk_id, chunk_text, chunk_size, 'sentence-aware'))

    return chunks

def sentence_aware_chunking(dataset):
    chunks_sentence = []
    for doc in dataset:
        doc_id = doc['doc_id']
        doc_title = doc['title']
        text = doc['text']

        chunks_sentence.extend(chunking_sentence_aware(doc_id, doc_title, text, max_chunk_size))
    
    return chunks_sentence

# c. Semantic chunking

threashold = 0.1
chunk_size_semantic = []
similarities = []

def chunking_semantic(doc_id, doc_title, text, max_chunk_size, threashold):

    # Get sentence list from the doc
    sentences = sent_tokenize(text)

    # chunks is array of dictionary 
    chunks = []

    # for generating each chunk of sentence to append to chunks, add the first sentence to the first chunk
    current_chunk = []

    # Total tokens of that chunk
    current_chunk_tokens = 0

    # indexing the chunk
    chunk_id = 0

    # save previous embedding 
    prev_embedding = None

    for i in range(len(sentences)):

        sentence = sentences[i]
        # Calculate token count for this sentence alone
        tokens = tokenizer.encode(sentence)

        # if a sentence token exceeds the max_chunk_size, we will have to split it into multiple chunks using fixed chunking
        if len(tokens) > max_chunk_size:
            start = 0
            while start < len(tokens):
                end = start + max_chunk_size
                # Get the tokens between start and end, chunk boundary
                chunk_tokens = tokens[start:end]
                # Get text from tokens. Decode
                chunk_text = tokenizer.decode(chunk_tokens)
                # Make CHUNK
                chunk_size =  len(chunk_tokens)
                chunk_size_semantic.append(chunk_size)
                chunks.append(CHUNK(doc_id, doc_title, chunk_id, chunk_text, chunk_size, 'semantic'))
                # To ensure overlap
                start += max_chunk_size - overlap
                chunk_id += 1
            
            prev_embedding = None  
            current_chunk = []
            current_chunk_tokens = 0
            continue  # Move to the next sentence after splitting the long sentence into chunks

        if prev_embedding is not None:

            embedding_1 = prev_embedding 
            embedding_2 = model.encode(sentence) 

            cosine_similarity = np.dot(embedding_1, embedding_2) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))

            # Append the cosine similarity to the golab array to find a optimal threashold
            similarities.append(cosine_similarity)

            # if similarity less than threshold, finish the current chunk and add to chunks
            # Check token count too
            if (cosine_similarity < threashold) or (current_chunk_tokens + len(tokens) > max_chunk_size):
                #print(i-1, i, cosine_similarity, threashold, len(cur_sentence.split()), len(current_chunk), max_chunk_size)

                # ---- Make CHUNK
                chunk_text = ' '.join(current_chunk)
                chunk_size = current_chunk_tokens
                chunk_size_semantic.append(chunk_size)
                #print(doc_id, chunk_size)
                chunks.append(CHUNK(doc_id, doc_title, chunk_id, chunk_text, chunk_size, 'semantic'))

                current_chunk = []
                current_chunk_tokens = 0
                chunk_id += 1
            
            prev_embedding = embedding_2
        else:
            # If there is no prev_embedding, then just add the cur sentence to the current_chunk (outside)and prev_embedding is just the cur sentence embedding
            prev_embedding = model.encode(sentence)

        # Append the sentence/the words from the sentences to current chunk
        current_chunk.append(sentence)
        current_chunk_tokens += len(tokens)

    # Outside for loop 
    # For last sentence
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunk_size = current_chunk_tokens
        chunk_size_semantic.append(chunk_size)
        chunks.append(CHUNK(doc_id, doc_title, chunk_id, chunk_text, chunk_size, 'semantic'))

    return chunks

def semantic_chunking(dataset):
    chunks_semantic = []
    for doc in dataset:
        doc_id = doc['doc_id']
        doc_title = doc['title']
        text = doc['text']

        chunks_semantic.extend(chunking_semantic(doc_id, doc_title, text, max_chunk_size, threashold))
    
    return chunks_semantic

# ----- Get or compute chunks and save to disk
import os
import pickle

def compute_chunks(chunking_fn, dataset, dataset_size, strategy_name):
    path = f"chunks/{dataset_size}/{strategy_name}.pkl"
    
    if os.path.exists(path):
        with open(path, 'rb') as f:
            #print("Load chunks from disk")
            return pickle.load(f)
    
    chunks = chunking_fn(dataset)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(chunks, f)
    
    return chunks

# ----- main
if __name__ == "__main__":

    chunks_fixed = fixed_chunking(sample)
    chunks_sentence = sentence_aware_chunking(sample)
    chunks_semantic = semantic_chunking(sample)

    # ----- Print fixed chunking
    print("---------- Fixed chunking")
    print("Avg chunk size : " + str(np.mean(chunk_size_fixed)))

    plt.figure()
    plt.hist(chunk_size_fixed, bins=20)
    plt.xlabel('Chunk size (tokens)')
    plt.ylabel('Count')
    plt.savefig('fixed_chunk_distribution.png')

    # ----- Print sentence aware chunking
    print("---------- Sentence aware chunking")
    print("Avg chunk size : " + str(np.mean(chunk_size_sentence)))

    plt.figure()
    plt.hist(chunk_size_sentence, bins=20)
    plt.xlabel('Chunk size (tokens)')
    plt.ylabel('Count')
    plt.savefig('sentence_chunk_distribution.png')

    # ----- Print Semantic aware chunking
    print("---------- Semantic aware chunking")
    print("Avg chunk size : " + str(np.mean(chunk_size_semantic)))
    

    plt.figure()
    plt.hist(chunk_size_semantic, bins=20)
    plt.xlabel('Chunk size (tokens)')
    plt.ylabel('Count')
    plt.savefig('semantic_chunk_distribution.png')

    plt.figure()
    plt.hist(similarities, bins=30)
    plt.xlabel('Adjacent sentence cosine similarity')
    plt.ylabel('Count')
    plt.savefig('similarity_histogram.png')
