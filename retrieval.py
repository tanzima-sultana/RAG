
import numpy as np

from constants import DENSE, BM25, HYBRID
from bm25_indexing import tokenize_chunk_text
from anthropic_api import anthropic_msg_api

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def RETRIEVED_OUTPUT(chunk_id, retrieved_chunk_ids, qus, context, generated_ans, ground_truth_ans, k, cost, latency):

    return {
        'chunk_id' : chunk_id,
        'retrieved_chunk_ids' : retrieved_chunk_ids,
        'qus': qus,
        'context': context,
        'generated_ans': generated_ans,
        'ground_truth_ans': ground_truth_ans,
        'k' : k,
        'cost' : cost,
        'latency' : latency
    }

# ----- Get ans using qus and context

def get_answer_from_qus_context(question, context):
    prompt = f"""You are a helpful assistant. Use the following context to answer the question. 
    If the answer is not contained within the context, say "I don't know."

    Context: {context}

    Question: {question}

    Answer:"""

    return anthropic_msg_api(prompt)

# ----- RRF : Reciprocal Rank Fushion ----- #

def reciprocal_rank_fusion(dense_chunk_ids, bm25_chunk_ids, k_const=60):
    scores = {}

    for rank, chunk_id in enumerate(dense_chunk_ids, start=1):
        #  scores.get(chunk_id, 0) - if the chunk_id already exists in teh scores map, meaning if a chunk appears in both the chunks type, add both
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k_const + rank)

    for rank, chunk_id in enumerate(bm25_chunk_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k_const + rank)

    # sort the cores map based on scores, high to low
    merged_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    merged_chunk_ids = [chunk_id for chunk_id, score in merged_sorted]

    return merged_chunk_ids

def retrieve_chunks(retrieval_type, chunk_type, eval_set, chunks, indexing, bm25, k):

    print("retrieval.py - Retieval :", retrieval_type, ", Chunk type : ", chunk_type)
    retrieved_outputs = []

    dense_indices = []
    if retrieval_type == DENSE or retrieval_type == HYBRID:
        questions = [item['question'] for item in eval_set]
        #print(len(questions))

        # 1. Query embedding
        query_embeddings = model.encode(questions, normalize_embeddings=True)
        print("query_embeddings shape - ", query_embeddings.shape)

        # 2. Find the closet k indices
        distances, dense_indices = indexing.search(query_embeddings, k)
        #print("indices shape - ", dense_indices.shape)

    # use this chunk_id to text map to retrieve chunk_text 
    # for dense and bm25, search_indices maintains same indexing of chunks, no need for mapping there
    chunk_id_to_text = {chunk['chunk_id']: chunk['chunk_text'] for chunk in chunks}

    for i, item in enumerate(eval_set):

        doc_id = item['doc_id']
        chunk_id = item['chunk_id']
        qus = item['question']
        ground_truth_ans = item['answer']

        # 3. Retrieve chunks_ids and chunk_texts
        retrieved_chunk_ids = []
        retrieved_chunk_texts =[]
        
        # ----- Hybrid
        if retrieval_type == HYBRID:
            # ----- For dense 
            dense_search_indices = dense_indices[i]
            # Retieve chunks
            dense_retrieved_chunk_ids = []
            dense_retrieved_chunk_texts =[]
            for j in dense_search_indices:
                dense_retrieved_chunk_ids.append(chunks[j]['chunk_id'])
                dense_retrieved_chunk_texts.append(chunks[j]['chunk_text'])

            # ----- For BM25
            tokenized_qus = tokenize_chunk_text(qus) 
            scores = bm25.get_scores(tokenized_qus) 
            bm25_search_indices = np.argsort(scores)[::-1][:k]
            # Retieve chunks
            bm25_retrieved_chunk_ids = []
            bm25_retrieved_chunk_texts =[]
            for j in bm25_search_indices:
                bm25_retrieved_chunk_ids.append(chunks[j]['chunk_id'])
                bm25_retrieved_chunk_texts.append(chunks[j]['chunk_text'])
            
            # RRF and slice at k
            retrieved_chunk_ids = reciprocal_rank_fusion(dense_retrieved_chunk_ids, bm25_retrieved_chunk_ids, k_const=60)[:k]
            retrieved_chunk_texts = [chunk_id_to_text[cid] for cid in retrieved_chunk_ids]

        # ----- Dense & BM25
        else:
            search_indices = []
            if retrieval_type == DENSE:
                search_indices = dense_indices[i]
            elif retrieval_type == BM25:
                tokenized_qus = tokenize_chunk_text(qus) # Tokenize qus
                scores = bm25.get_scores(tokenized_qus)  # one score per chunk, same order as chunks list
                search_indices = np.argsort(scores)[::-1][:k]  # descending order, take top k

            # Retieve chunks
            for j in search_indices:
                retrieved_chunk_ids.append(chunks[j]['chunk_id'])
                retrieved_chunk_texts.append(chunks[j]['chunk_text'])
        

        # Generate answer
        context = ' '.join(retrieved_chunk_texts) 
        api_response = get_answer_from_qus_context(qus, context)
        cost = api_response['cost']
        latency = api_response['latency']
        generated_ans = api_response['response']

        retrieved_outputs.append(RETRIEVED_OUTPUT(chunk_id, retrieved_chunk_ids, qus, context, generated_ans, ground_truth_ans, k, cost, latency))
    
    return retrieved_outputs
