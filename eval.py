import os
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
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

def EVAL_METRICES(k, recall, ans_faithfullness, ans_relevancy, cost, latency):

    return {
        'k': k,
        'recall': recall,
        'ans_faithfullness': ans_faithfullness,
        'ans_relevancy': ans_relevancy,
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


def generate_question_for_chunk(chunk):
    prompt = f"""Here is a passage of text:

    {chunk['chunk_text']}

    Write one factual question whose answer is directly contained in this passage. 
    The question should be phrased naturally, the way a real user would ask it — not by echoing the passage's exact wording. 
    Then give the answer, using the passage's information.

    Respond in this exact format:
    Question: <question>
    Answer: <answer>"""

    api_response = anthropic_msg_api(prompt)
    response = api_response['response']
    
    return response.content[0].text

def generate_questions(chunks, no_of_questions, min_chunk_size): 
    chunks = sample_chunks_for_eval(chunks, no_of_questions, min_chunk_size)
    questions = []
    for chunk in chunks:
        qa = generate_question_for_chunk(chunk)
        parsed = parse_qa_response(qa)
        qus, ans = parsed['question'], parsed['answer']
        questions.append(EVAL_QUS(chunk['doc_id'], chunk['chunk_id'], qus, ans))      

    # Save the generated questions to a JSON file
    path = f"eval/QA_{no_of_questions}/eval_set.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(questions, f, indent=2)

    return questions 

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

# ----- Evaluate

def eval_computation(chunks, indexing, k):
    
    ''''
    # ----- Generated qus
    no_of_questions = 30
    min_chunk_size = 100

    path1 = f"eval/QA_30/eval_set.json"
    if os.path.exists(path1):
        with open(path1, 'r') as f:
            generated_qus = json.load(f)
    else:
        generated_qus = generate_questions(chunks, no_of_questions, min_chunk_size)

    #print(generated_qus)

    # ----- User defined qus

    user_defined_qus = None 
    path2 = f"eval/QA_20_User/eval_set.json"
    if os.path.exists(path2):
        with open(path2, 'r') as f:
            user_defined_qus = json.load(f)
    
    #print(user_defined_qus)
    '''

    # ----- Eval qus and ans

    #eval_set = {doc_id, chunk_id, question, answer}

    eval_set = None
    path = f"eval/QA_50/eval_set.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            eval_set = json.load(f)

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
        
        # 6. Answer faithfulness & relevancy
        score_faithfulness, api_response2 = judge_faithfulness(generated_ans, context)
        cost2 = api_response2['cost']
        latency2 = api_response2['latency']

        score_relevancy, api_response3 = judge_relevancy(generated_ans, qus)
        cost3 = api_response3['cost']
        latency3 = api_response3['latency']

        answer_faithfulness.append(score_faithfulness)
        answer_relevancy.append(score_relevancy)

        total_cost += cost1 + cost2 + cost3
        total_latency += latency1 + latency2 + latency3

        eval_metrices.append(EVAL_METRICES(k, recall, score_faithfulness, score_relevancy, cost1+cost2+cost3, latency1+ latency2 + latency3))

        print("----- i : ", i)
        print(" Qus : ", qus)
        print("Generated ans : ", generated_ans)
        print("Ground truth ans : ", ground_truth_ans)
        print("Matrices : ", recall, score_faithfulness, score_relevancy)
        print("Cost : ", cost1, cost2, cost3)
        print("Latency : ", latency1, latency2, latency3)
    
    print("total cost : ", total_cost, ", total latency : ", total_latency)

    recall = total_recall / len(questions)
    avg_ans_faithfulness = sum(answer_faithfulness)/len(answer_faithfulness)
    avg_ans_relevancy = sum(answer_relevancy)/len(answer_relevancy)
    print("recall, avg ans faithfulness, avg ans releavncy : ", recall, avg_ans_faithfulness, avg_ans_relevancy)

    return eval_results, eval_metrices, recall, avg_ans_faithfulness, avg_ans_relevancy, total_cost, total_latency


def evaluate(dataset_size, strategy, chunks, indexing, k):
    print("Evaluate for : ", strategy)
    
    eval_results, eval_metrices, recall, avg_ans_faithfulness, avg_ans_relevancy, total_cost, total_latency = eval_computation(chunks, indexing, k)

    eval_summary = {
        'dataset_size': dataset_size,
        'strategy': strategy,
        'k': k,
        'num_questions': len(eval_metrices),
        'recall': recall,
        'avg_faithfulness': avg_ans_faithfulness,
        'avg_relevancy': avg_ans_relevancy,
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


    


