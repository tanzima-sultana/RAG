from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np

from config import SEED
import random
import os
import json

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

def EVAL_METRICES(k, recall, mrr, precision_at_k, sas_score,
                    score_faithfulness, score_relevancy, score_llm_correctness, cost, latency):

    return {
        'k': k,
        'recall': recall,
        'mrr': mrr,
        'precision_at_k': precision_at_k,
        'sas_score': sas_score,
        'score_faithfulness': score_faithfulness,
        'score_relevancy': score_relevancy,
        'score_llm_correctness': score_llm_correctness,
        'cost': cost,
        'latency': latency
    }

def EVAL_RESULT(qus, context, generated_ans, ground_truth_ans):

    return {
        'qus': qus,
        'context': context,
        'generated_ans': generated_ans,
        'ground_truth_ans': ground_truth_ans
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
    lines = response_text.strip().split('\n')
    question = lines[0].replace('Question:', '').strip()
    answer = lines[1].replace('Answer:', '').strip()
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
    

def build_generated_eval_set(strategy, dataset, dataset_size, chunks, min_chunk_size, no_of_qus): 

    path = f"eval/QA_{dataset_size}/{strategy}_{no_of_qus}/eval_set.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)

    sample_chunks = sample_chunks_for_eval(chunks, no_of_qus, min_chunk_size)

    eval_set = []

    for chunk in sample_chunks:
        qa = generate_qa_from_chunk(chunk)
        parsed = parse_qa_response(qa)
        qus, ans = parsed['question'], parsed['answer']
        eval_set.append(EVAL_QUS(chunk['doc_id'], chunk['chunk_id'], qus, ans))  

    # Save the generated questions to a JSON file
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(eval_set, f, indent=2)

    return eval_set 


# ----- Get ans using qus and context

def get_answer_from_qus_context(question, context):
    prompt = f"""You are a helpful assistant. Use the following context to answer the question. 
    If the answer is not contained within the context, say "I don't know."

    Context: {context}

    Question: {question}

    Answer:"""

    return anthropic_msg_api(prompt)
    

# ----- Answer Faithfulness 
def judge_faithfulness(generated_answer, context):
    prompt = f"""You are a helpful assistant. Judge whether the following answer is faithful to the provided context. 
    If the answer is fully supported by the context, respond with "Faithful". 
    If the answer contains information not present in the context, respond with "Not Faithful".

    Context: {context}

    Answer: {generated_answer}

    Is the answer faithful to the context?"""
    
    api_response = anthropic_msg_api(prompt)
    judgement = api_response['response']

    if "not faithful" in judgement.lower():
        return 0, api_response
    elif "faithful" in judgement.lower():
        return 1, api_response
    else:
        return 0, api_response  

# ----- Answer Relevancy
def judge_relevancy(generated_answer, question):
    prompt = f"""You are a helpful assistant. Judge whether the following answer is relevant to the provided question. 
    If the answer is relevant to the question, respond with "Relevant". 
    If the answer is not relevant to the question, respond with "Not Relevant".

    Question: {question}

    Answer: {generated_answer}

    Is the answer relevant to the question?"""
    
    api_response = anthropic_msg_api(prompt)
    judgement = api_response['response']

    if "not relevant" in judgement.lower():
        return 0, api_response
    elif "relevant" in judgement.lower():
        return 1, api_response
    else:
        return 0, api_response  

# ----- Answer LLM correectness
def judge_llm_correctness(generated_answer, ground_truth_ans):
    prompt = f"""You are a helpful assistant. Judge whether the following generated answer matches the ground truth answer in meaning, even if phrased differently. 
    If the generated answer captures the key information in the ground truth, respond with "Correct". 
    If the generated answer is missing key information or contradicts the ground truth, respond with "Incorrect".

    Ground truth answer: {ground_truth_ans}

    Generated answer: {generated_answer}

    Does the generated answer match the ground truth?"""
    
    api_response = anthropic_msg_api(prompt)
    judgement = api_response['response']

    if "incorrect" in judgement.lower():
        return 0, api_response
    elif "correct" in judgement.lower():
        return 1, api_response
    else:
        return 0, api_response

# ----- MRR ----- #
def compute_mrr(chunk_id, retrieved_chunk_ids):
    if chunk_id in retrieved_chunk_ids:
        rank = retrieved_chunk_ids.index(chunk_id) + 1
        return 1 / rank
    return 0

# -----  Precision ----- #
def compute_precision_at_k(chunk_id, retrieved_chunk_ids, k):
    return (1 if chunk_id in retrieved_chunk_ids else 0) / k

# ----- Cosine Similarity ----- #
def compute_semantic_similarity(generated_ans, ground_truth_ans):
    emb1 = model.encode(generated_ans, normalize_embeddings=True)
    emb2 = model.encode(ground_truth_ans, normalize_embeddings=True)
    return float(np.dot(emb1, emb2))

# ----- Evaluate

def evaluate(strategy, dataset, dataset_size, chunks, indexing, k,
             use_faithfulness=False, use_relevancy=False, use_llm_correctness=False):

    print("Evaluate for : ", strategy)

    # ---- Get evat_set

    #eval_set = build_nq_eval_set(dataset, dataset_size, chunks)
    min_chunk_size = 100
    no_of_qus = 50
    eval_set = build_generated_eval_set(strategy, dataset, dataset_size, chunks, min_chunk_size, no_of_qus)
    #print(len(eval_set), eval_set)

    doc_ids = [item['doc_id'] for item in eval_set]
    chunks_ids = [item['chunk_id'] for item in eval_set]
    questions = [item['question'] for item in eval_set]
    answers = [item['answer'] for item in eval_set]

    print(len(questions))

    # ----- Recall, Answer Faithfulness & Relevancy

    eval_results = []
    eval_metrices = []

    
    answer_faithfulness = []
    answer_relevancy = []
    answer_llm_correctness = []

    recall = 0

    total_recall = 0
    total_cost = 0
    total_latency = 0

    # 1. Query embedding
    query_embeddings = model.encode(questions, normalize_embeddings=True)
    print("query_embeddings shape - ", query_embeddings.shape)

    # 2. Find the closet k indices
    distances, indices = indexing.search(query_embeddings, k)
    print("indices shape - ", indices.shape)
    
    for i in range(len(questions)):

        chunk_id = chunks_ids[i]
        qus = questions[i]
        ground_truth_ans = answers[i]

        # 3. Retrieve chunks_ids and chunk_texts
        retrieved_chunk_ids = []
        retrieved_chunk_texts =[]
        for j in indices[i]:
            retrieved_chunk_ids.append(chunks[j]['chunk_id'])
            retrieved_chunk_texts.append(chunks[j]['chunk_text'])
        
        # 4. Generate answer
        context = ' '.join(retrieved_chunk_texts) 
        api_response1 = get_answer_from_qus_context(qus, context)
        cost1 = api_response1['cost']
        latency1 = api_response1['latency']
        generated_ans = api_response1['response']

        eval_results.append(EVAL_RESULT(qus, context, generated_ans, ground_truth_ans))

        # 5. Recall
        if chunk_id in retrieved_chunk_ids:
            recall = 1
            total_recall += 1
        else:
            recall = 0

        # 6. MRR, precision and SAS score
        mrr = compute_mrr(chunk_id, retrieved_chunk_ids)
        precision_at_k = compute_precision_at_k(chunk_id, retrieved_chunk_ids, k)
        sas_score = compute_semantic_similarity(generated_ans, ground_truth_ans)
        
        # 7. Answer faithfulness
        score_faithfulness = 0
        cost2 = 0
        latency2 = 0
        if use_faithfulness:
            score_faithfulness, api_response2 = judge_faithfulness(generated_ans, context)
            cost2 = api_response2['cost']
            latency2 = api_response2['latency']

        # 8. Answer relevancy
        score_relevancy = 0
        cost3 = 0
        latency3 = 0
        if use_relevancy:
            score_relevancy, api_response3 = judge_relevancy(generated_ans, qus)
            cost3 = api_response3['cost']
            latency3 = api_response3['latency']
        
        # 9. Answer LLM correctness
        score_llm_correctness = 0
        cost4 = 0
        latency4 = 0
        if use_llm_correctness:
            score_llm_correctness, api_response4 = judge_llm_correctness(generated_ans, ground_truth_ans)
            cost4 = api_response4['cost']
            latency4 = api_response4['latency']

        answer_faithfulness.append(score_faithfulness)
        answer_relevancy.append(score_relevancy)
        answer_llm_correctness.append(score_llm_correctness)

        total_cost += cost1 + cost2 + cost3 + cost4
        total_latency += latency1 + latency2 + latency3 + latency4

        eval_metrices.append(EVAL_METRICES(k, recall, 
                                           mrr, precision_at_k, sas_score,
                                           score_faithfulness, score_relevancy, score_llm_correctness,
                                           cost1+cost2+cost3+cost4, latency1+ latency2 + latency3+ latency4))

        print("----- i : ", i)
        print(" Qus : ", qus)
        print("Generated ans : ", generated_ans)
        print("Ground truth ans : ", ground_truth_ans)
        print("Matrices : recall, MRR, precision_at_k, sas_score : ", recall, mrr, precision_at_k, sas_score)
        print("Matrices : score_faithfulness, score_relevancy, score_llm_correctness : ", score_faithfulness, score_relevancy, score_llm_correctness)
        print("Cost : ", cost1, cost2, cost3)
        print("Latency : ", latency1, latency2, latency3)
    
    print("total cost : ", total_cost, ", total latency : ", total_latency)

    recall = total_recall / len(questions)
    avg_ans_faithfulness = sum(answer_faithfulness)/len(answer_faithfulness)
    avg_ans_relevancy = sum(answer_relevancy)/len(answer_relevancy)
    avg_ans_lmm_correctness = sum(answer_llm_correctness) / len(answer_llm_correctness)
    print("recall, avg ans faithfulness, avg ans releavncy, avg_ans_lmm_correctness : ", 
          recall, avg_ans_faithfulness, avg_ans_relevancy, avg_ans_lmm_correctness)

    eval_summary = {
        'dataset_size': dataset_size,
        'strategy': strategy,
        'k': k,
        'num_questions': len(eval_metrices),
        'recall': recall,
        'avg_faithfulness': avg_ans_faithfulness,
        'avg_relevancy': avg_ans_relevancy,
        'avg_ans_lmm_correctness' : avg_ans_lmm_correctness,
        'total_cost': total_cost,
        'total_latency': total_latency,
    }

    out_dir = f"eval/results/{dataset_size}"
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/{strategy}_summary.json", 'w') as f:
        json.dump(eval_summary, f, indent=2)
    
    with open(f"{out_dir}/{strategy}_details.json", 'w') as f:
        json.dump({'eval_results': eval_results, 'eval_metrices': eval_metrices}, f, indent=2)

    return eval_summary

    

    


