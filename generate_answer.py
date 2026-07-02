from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt

import os
import pickle

from anthropic_api import anthropic_msg_api

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_answer_from_qus_context(question, context):
    prompt = f"""You are a helpful assistant. Use the following context to answer the question. 
    If the answer is not contained within the context, say "I don't know."

    Context: {context}

    Question: {question}

    Answer:"""

    return anthropic_msg_api(prompt)

def get_single_answer(chunks, indexing, question, k):

    # 1. Query embedding, make question as [] list to process
    query_embedding = model.encode([question], normalize_embeddings=True)

    # 2. Retrieve context 
    # indexing.search can process multiple query in parallel. So it expects 2D querry_embedding each for one query of 384 dim
    # returns a 2D where each row is distances list and indices list 
    # distances contains distance between query and the chunks, indices contains index from indexing which aligns with chunks index
    
    # For FlatIndexIP - distance is inner dot product/cosine similarity due to normalized embeddings value
    context = []
    distances, indices = indexing.search(query_embedding, k)

    #print(distances, indices)
    
    retrieved_chunks =[]
    for i in indices[0]:
        retrieved_chunks.append(chunks[i]['chunk_text'])
    context = ' '.join(retrieved_chunks) 

    # 3. Generate answer using retrieved context and the qus. *** This answer is given to user
    api_response = get_answer_from_qus_context(question, context)
    cost = api_response['cost']
    latency = api_response['latency']
    generated_ans = api_response['response']

    #print( cost, latency)

    return generated_ans
