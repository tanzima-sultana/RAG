
import numpy as np
import json
import os
from anthropic_api import anthropic_msg_api

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

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
    
# ----- Recall ----- #
# multi-chunk : no of relevant chunks in top k / total number of questions
# one chunk : r = 1 if that one chunk is in top k. Otherwise 0. sum(r) / total number of questions

def compute_recall_single_chunk(chunk_id, retrieved_chunk_ids):
    return 1 if chunk_id in retrieved_chunk_ids else 0

def compute_recall_multi_chunk(relevant_chunk_ids, retrieved_chunk_ids):
    relevant_set = set(relevant_chunk_ids)
    retrieved_set = set(retrieved_chunk_ids)
    common_set = relevant_set & retrieved_set
    return len(common_set) / len(relevant_set)

# -----  Precision ----- #
# no of relevant chunks in top k / k
# one chunk: p = 1 if that one chunk is in top k. Otherwise 0. p / k

def compute_precision_at_k_single_chunk(chunk_id, retrieved_chunk_ids, k):
    return (1 if chunk_id in retrieved_chunk_ids else 0) / k

def compute_precision_at_k_multi_chunk(relevant_chunk_ids, retrieved_chunk_ids, k):
    relevant_set = set(relevant_chunk_ids)
    retrieved_set = set(retrieved_chunk_ids)
    common_set = relevant_set & retrieved_set
    return len(common_set) / k

# ----- MRR ----- #
# Mean Reciprocal Rank
# inverse of rank position (index+1) of the first correct answer. Position of chunk_id in the retrieved chunk_ids

def compute_mrr(chunk_id, retrieved_chunk_ids):
    if chunk_id in retrieved_chunk_ids:
        rank = retrieved_chunk_ids.index(chunk_id) + 1
        return 1 / rank
    return 0

# ----- Cosine Similarity ----- #
# dot product of generated ans and ground truth ans

def compute_semantic_similarity(generated_ans, ground_truth_ans):
    emb1 = model.encode(generated_ans, normalize_embeddings=True)
    emb2 = model.encode(ground_truth_ans, normalize_embeddings=True)
    return float(np.dot(emb1, emb2))

def evaluate(retrieval_type, chunk_type, dataset_size, retrieved_output,
            use_faithfulness=False, use_relevancy=False, use_llm_correctness=False):
    
    print("evaluation.py - Retieval :", retrieval_type, ", Chunk type : ", chunk_type)

    eval_metrices = []

    total_recall = 0
    total_cost = 0
    total_latency = 0

    answer_faithfulness = []
    answer_relevancy = []
    answer_llm_correctness = []

    for output in retrieved_output:
        chunk_id = output['chunk_id']
        retrieved_chunk_ids = output['retrieved_chunk_ids']
        qus = output['qus']
        context = output['context']
        generated_ans = output['generated_ans']
        ground_truth_ans = output['ground_truth_ans']
        k = output['k']
        cost1 = output['cost']
        latency1 = output['latency']

        # 1. Recall
        recall = compute_recall_single_chunk(chunk_id, retrieved_chunk_ids)

        # 2. Precision
        precision_at_k = compute_precision_at_k_single_chunk(chunk_id, retrieved_chunk_ids, k)

        # 3. MRR and SAS score
        mrr = compute_mrr(chunk_id, retrieved_chunk_ids)
        sas_score = compute_semantic_similarity(generated_ans, ground_truth_ans)

        # 4. Answer faithfulness
        score_faithfulness = 0
        cost2 = 0
        latency2 = 0
        if use_faithfulness:
            score_faithfulness, api_response2 = judge_faithfulness(generated_ans, context)
            cost2 = api_response2['cost']
            latency2 = api_response2['latency']
        
        # 5. Answer relevancy
        score_relevancy = 0
        cost3 = 0
        latency3 = 0
        if use_relevancy:
            score_relevancy, api_response3 = judge_relevancy(generated_ans, qus)
            cost3 = api_response3['cost']
            latency3 = api_response3['latency']

        # 6. Answer LLM correctness
        score_llm_correctness = 0
        cost4 = 0
        latency4 = 0
        if use_llm_correctness:
            score_llm_correctness, api_response4 = judge_llm_correctness(generated_ans, ground_truth_ans)
            cost4 = api_response4['cost']
            latency4 = api_response4['latency']

        cost = cost1 + cost2 + cost3 + cost4 
        latency = latency1 + latency2 + latency3 + latency4

        eval_metrices.append(EVAL_METRICES(k, recall, mrr, precision_at_k, sas_score, 
                                    score_faithfulness, score_relevancy, score_llm_correctness, cost, latency))

        print(" Qus : ", qus)
        print("Generated ans : ", generated_ans)
        print("Ground truth ans : ", ground_truth_ans)
        print("Matrices : recall, MRR, precision_at_k, sas_score : ", recall, mrr, precision_at_k, sas_score)
        print("Matrices : score_faithfulness, score_relevancy, score_llm_correctness : ", score_faithfulness, score_relevancy, score_llm_correctness)
        print("Cost : ", cost)
        print("Latency : ", latency)

        total_recall += recall
        total_cost += cost
        total_latency += latency
        answer_faithfulness.append(score_faithfulness)
        answer_relevancy.append(score_relevancy)
        answer_llm_correctness.append(score_llm_correctness)
    

    print("total cost : ", total_cost, ", total latency : ", total_latency)

    avg_ans_faithfulness = sum(answer_faithfulness)/len(answer_faithfulness)
    avg_ans_relevancy = sum(answer_relevancy)/len(answer_relevancy)
    avg_ans_lmm_correctness = sum(answer_llm_correctness) / len(answer_llm_correctness)
    print("recall, avg ans faithfulness, avg ans releavncy, avg_ans_lmm_correctness : ", 
          total_recall / len(retrieved_output), avg_ans_faithfulness, avg_ans_relevancy, avg_ans_lmm_correctness)

    eval_summary = {
        'dataset_size': dataset_size,
        'retrieval_type': retrieval_type,
        'chunk_type': chunk_type,
        'k': k,
        'num_questions': len(retrieved_output),
        'recall': total_recall / len(retrieved_output),
        'avg_faithfulness': avg_ans_faithfulness,
        'avg_relevancy': avg_ans_relevancy,
        'avg_ans_lmm_correctness' : avg_ans_lmm_correctness,
        'total_cost': total_cost,
        'total_latency': total_latency,
    }

    out_dir = f"eval/results/{dataset_size}"
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/{retrieval_type}_{chunk_type}_{k}_summary.json", 'w') as f:
        json.dump(eval_summary, f, indent=2)
    
    with open(f"{out_dir}/{retrieval_type}_{chunk_type}_{k}_details.json", 'w') as f:
        json.dump({'retrieved_output': retrieved_output, 'eval_metrices': eval_metrices}, f, indent=2)

    return eval_summary