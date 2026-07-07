from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
import random
import os
import json


from constants import SEED
from anthropic_api import anthropic_msg_api

model = SentenceTransformer('all-MiniLM-L6-v2')

# ----- MACRO

def EVAL_QUS(doc_id, chunk_id, question, answer):

    return {
        'doc_id': doc_id,
        'chunk_id': chunk_id,
        'question': question,
        'answer': answer
    }

 # ----- NQ questions ----- #

def normalize(text):
    return ' '.join(text.lower().split())

def word_overlap_ratio(text1, text2):
    words1 = set(normalize(text1).split())
    words2 = set(normalize(text2).split())
    if not words1:
        return 0
    return len(words1 & words2) / len(words1)

def find_matching_doc(answer_text, dataset, threshold=0.5):
    best_doc = None
    best_score = 0
    for doc in dataset:
        score = word_overlap_ratio(answer_text, doc['text'])
        if score > best_score:
            best_score = score
            best_doc = doc
    if best_score >= threshold:
        return best_doc
    return None

def find_matching_chunk(answer_text, chunks_for_doc, threshold=0.5):
    best_chunk = None
    best_score = 0
    for chunk in chunks_for_doc:
        score = word_overlap_ratio(answer_text, chunk['chunk_text'])
        if score > best_score:
            best_score = score
            best_chunk = chunk
    if best_score >= threshold:
        return best_chunk
    return None

def build_nq_eval_set(dataset, dataset_size, chunks):

    # If eval_set already exists
    path = f"eval/NQ_{dataset_size}/eval_set.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
        
    nq_raw = load_dataset("sentence-transformers/natural-questions", split="train")
    nq_sample = nq_raw.shuffle(seed=SEED).select(range(2000))

    eval_set = []
    for example in nq_sample:
        question = example['query']
        answer = example['answer']

        # First find the doc that has the best overlapping score against the answer
        matched_doc = find_matching_doc(answer, dataset)
        if matched_doc is None:
            continue
        
        # From the docs, find the best chunk
        doc_chunks = [c for c in chunks if c['doc_id'] == matched_doc['doc_id']]
        matched_chunk = find_matching_chunk(answer, doc_chunks)
        if matched_chunk is None:
            continue

        eval_set.append(EVAL_QUS(matched_doc['doc_id'], matched_chunk['chunk_id'], question, answer))
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(eval_set, f, indent=2)

    return eval_set

# ----- Auto generate qus 

def sample_chunks_for_eval(chunks, n_samples, min_chunk_size):
    substantive_chunks = [c for c in chunks if c['chunk_size'] >= min_chunk_size]
    
    random.seed(SEED)
    sampled = random.sample(substantive_chunks, n_samples)
    
    return sampled

def parse_qa_response(response_text):
    question = None
    answer = None

    for line in response_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('Question:'):
            question = line.replace('Question:', '', 1).strip()
        elif line.startswith('Answer:'):
            answer = line.replace('Answer:', '', 1).strip()
        elif answer is not None:
            # multi-line answer continuation
            answer += ' ' + line

    if question is None or answer is None:
        raise ValueError(f"Could not parse Q/A from response:\n{response_text}")

    return {'question': question, 'answer': answer}


def generate_qa_from_chunk(chunk):
    prompt = f"""Here is a passage of text:

    {chunk['chunk_text']}

    Write one factual question whose answer is directly contained in this passage. 
    The question should be phrased naturally, the way a real user would ask it — not by echoing the passage's exact wording. 
    Then give the answer, using the passage's information.

    Respond in this exact format:
    Question: <question>
    Answer: <answer>"""

    api_response = anthropic_msg_api(prompt)
    return api_response['response']
    

def build_eval_set(strategy, dataset, dataset_size, chunks, min_chunk_size, no_of_qus): 

    path = f"eval/QA_{dataset_size}/{strategy}_{no_of_qus}/eval_set.json"
    
    eval_set = []
    done_chunk_ids = set()

    if os.path.exists(path):
        with open(path, 'r') as f:
            eval_set = json.load(f)
        if len(eval_set) >= no_of_qus:
            return eval_set
        done_chunk_ids = {e['chunk_id'] for e in eval_set}

    sample_chunks = sample_chunks_for_eval(chunks, no_of_qus, min_chunk_size)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    for chunk in sample_chunks:
        if chunk['chunk_id'] in done_chunk_ids:
            continue

        qa = generate_qa_from_chunk(chunk)
        parsed = parse_qa_response(qa)
        qus, ans = parsed['question'], parsed['answer']
        eval_set.append(EVAL_QUS(chunk['doc_id'], chunk['chunk_id'], qus, ans))

        with open(path, 'w') as f:
            json.dump(eval_set, f, indent=2)

    return eval_set

