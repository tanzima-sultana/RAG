
import numpy as np
import json
import os
import re
import boto3 
import time
from src.anthropic_api import AnthropicAPI
from src.dist import s3_utills

from config import S3_BUCKET
from constants import LOCAL, AWS
from sentence_transformers import SentenceTransformer

class Evaluation:
    def __init__(self, mode, dataset_size, model_name, use_llm_judge):
        self.mode = mode
        self.dataset_size = dataset_size
        self.model_name = model_name
        self.use_llm_judge = use_llm_judge

        self.model = SentenceTransformer(model_name)

        self.anthropic = AnthropicAPI(anthropic_model="claude-sonnet-4-6", max_tokens=4000)

        self.path = f"evals/{dataset_size}"
        if self.mode == AWS:
            self.path = f"s3://{S3_BUCKET}/" + self.path
    
    def EVAL_METRICES(self, k, recall, mrr, precision_at_k, sas_score,
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
    def get_answer_from_qus_context(self, question, context):
        prompt = f"""You are a helpful assistant. Use the following context to answer the question. 
        If the answer is not contained within the context, say "I don't know."

        Context: {context}

        Question: {question}

        Answer:"""

        return self.anthropic.anthropic_msg_api(prompt)
    
    '''
    # ----- Answer Faithfulness 
    def judge_faithfulness(self, generated_answer, context):
        prompt = f"""You are a helpful assistant. Judge whether the following answer is faithful to the provided context. 
        If the answer is fully supported by the context, respond with "Faithful". 
        If the answer contains information not present in the context, respond with "Not Faithful".

        Context: {context}

        Answer: {generated_answer}

        Is the answer faithful to the context?"""
        
        api_response = self.anthropic.anthropic_msg_api(prompt)
        judgement = api_response['response']

        if "not faithful" in judgement.lower():
            return 0, api_response
        elif "faithful" in judgement.lower():
            return 1, api_response
        else:
            return 0, api_response  
    
    # ----- Answer Relevancy
    def judge_relevancy(self, generated_answer, question):
        prompt = f"""You are a helpful assistant. Judge whether the following answer is relevant to the provided question. 
        If the answer is relevant to the question, respond with "Relevant". 
        If the answer is not relevant to the question, respond with "Not Relevant".

        Question: {question}

        Answer: {generated_answer}

        Is the answer relevant to the question?"""
        
        api_response = self.anthropic.anthropic_msg_api(prompt)
        judgement = api_response['response']

        if "not relevant" in judgement.lower():
            return 0, api_response
        elif "relevant" in judgement.lower():
            return 1, api_response
        else:
            return 0, api_response  

    # ----- Answer LLM correectness
    def judge_llm_correctness(self, generated_answer, ground_truth_ans):
        prompt = f"""You are a helpful assistant. Judge whether the following generated answer matches the ground truth answer in meaning, even if phrased differently. 
        If the generated answer captures the key information in the ground truth, respond with "Correct". 
        If the generated answer is missing key information or contradicts the ground truth, respond with "Incorrect".

        Ground truth answer: {ground_truth_ans}

        Generated answer: {generated_answer}

        Does the generated answer match the ground truth?"""
        
        api_response = self.anthropic.anthropic_msg_api(prompt)
        judgement = api_response['response']

        if "incorrect" in judgement.lower():
            return 0, api_response
        elif "correct" in judgement.lower():
            return 1, api_response
        else:
            return 0, api_response
    '''
    # ------ LLM Judge Faithfulness, Relevancy and Correctness 

    def _extract_batch_score(self, parsed, i):
        entry = parsed.get(str(i))  # JSON keys are always strings, even if chunk_id is an int in your data
        if not isinstance(entry, dict):
            return 0.0, 0.0, 0.0
        try:
            return (
                float(entry.get("faithfulness", 0.0)),
                float(entry.get("relevancy", 0.0)),
                float(entry.get("correctness", 0.0)),
            )
        except (TypeError, ValueError):
            return 0.0, 0.0, 0.0

    def _regex_fallback_batch_score(self, text, i):
        pattern = (
            rf'"{re.escape(str(i))}"\s*:\s*{{\s*'
            rf'"faithfulness"\s*:\s*([\d.]+)\s*,\s*'
            rf'"relevancy"\s*:\s*([\d.]+)\s*,\s*'
            rf'"correctness"\s*:\s*([\d.]+)'
        )
        match = re.search(pattern, text)
        if not match:
            return 0.0, 0.0, 0.0
        return float(match.group(1)), float(match.group(2)), float(match.group(3))

    def parse(self, judgement_text, batch):
        cleaned = re.sub(r"^```(json)?|```$", "", judgement_text.strip(), flags=re.MULTILINE).strip()

        try:
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, AttributeError):
            print(f"[WARN] Batch judge JSON parse failed entirely. Raw: {judgement_text[:200]}")
            return {}

        results = {}
        for key, entry in parsed.items():
            i = int(key)
            results[i] = self._extract_batch_score(parsed, i)
        return results

    def get_llm_judgement_prompt(self, block):
        return f"""You are evaluating multiple RAG system outputs, each identified by a i.
            For EACH item, score three INDEPENDENT dimensions. Do not let one item's
            judgment influence another item, and do not let one dimension bleed into
            another within the same item — a hallucinated answer can still be "relevant."

            Each item below is tagged [i: n]. Use that same i as the key
            in your response.

            ITEMS:
            {block}

            For each item, score (0.0-1.0):
            1. FAITHFULNESS: Is generated_ans fully supported by context, with no claims
            absent from it? Ignore ground_truth_ans for this dimension.
            2. RELEVANCY: Does generated_ans directly address qus? Ignore context and
            ground_truth_ans for this dimension — only judge topical fit.
            3. CORRECTNESS: Does generated_ans capture the key information in
            ground_truth_ans, even if phrased differently? Ignore context for this dimension.

            Respond with ONLY this JSON — one key per i shown above, no other text,
            no markdown fences:
            {{"<i>": {{"faithfulness": <float>, "relevancy": <float>, "correctness": <float>}}, "<i>": {{...}}, ...}}

            Every i shown in ITEMS must appear as a key in your response, even if
            you have to score it 0.0 — do not skip or merge items."""

    def get_batch_llm_judgement(self, batch, start):

        block = ""

        for i, item in enumerate(batch):
            qus = item['qus']
            context = item['context']
            generated_ans = item['generated_ans']
            ground_truth_ans = item['ground_truth_ans']

            block += f"\n[i: {start + i}]\n qus: {qus}\n context: {context}\n generated_ans: {generated_ans}\n ground_truth_ans: {ground_truth_ans}\n"
        
        prompt = self.get_llm_judgement_prompt(block)
        api_response = self.anthropic.anthropic_msg_api(prompt)
        return api_response

    def _doc_id_from_chunk_id(self, chunk_id):
        return chunk_id.rsplit('_', 1)[0]

    # ----- Recall ----- #
    # multi-chunk : no of relevant chunks in top k / total number of questions
    # one chunk : r = 1 if that one chunk is in top k. Otherwise 0. sum(r) / total number of questions
    
    def compute_recall_single_chunk(self, gt_chunk_id, retrieved_chunk_ids):
        gt_doc = self._doc_id_from_chunk_id(gt_chunk_id)
        retrieved_docs = [self._doc_id_from_chunk_id(cid) for cid in retrieved_chunk_ids]
        return 1 if gt_doc in retrieved_docs else 0

    # -----  Precision ----- #
    # no of relevant chunks in top k / k
    # one chunk: p = 1 if that one chunk is in top k. Otherwise 0. p / k
    
    def compute_precision_at_k_single_chunk(self, gt_chunk_id, retrieved_chunk_ids, k):
        gt_doc = self._doc_id_from_chunk_id(gt_chunk_id)

        hits = 0
        for cid in retrieved_chunk_ids:
            cid_doc = self._doc_id_from_chunk_id(cid)
            if cid_doc == gt_doc:
                hits = hits + 1

        return hits / k
    
    # ----- MRR ----- #
    # Mean Reciprocal Rank
    # inverse of rank position (index+1) of the first correct answer. Position of chunk_id in the retrieved chunk_ids
    
    def compute_mrr(self, gt_chunk_id, retrieved_chunk_ids):
        gt_doc = self._doc_id_from_chunk_id(gt_chunk_id)
        for rank, cid in enumerate(retrieved_chunk_ids, start=1):
            if self._doc_id_from_chunk_id(cid) == gt_doc:
                return 1 / rank
        return 0

    # ----- Cosine Similarity ----- #
    # dot product of generated ans and ground truth ans

    def compute_semantic_similarity(self, generated_ans, ground_truth_ans):
        emb1 = self.model.encode(generated_ans, normalize_embeddings=True)
        emb2 = self.model.encode(ground_truth_ans, normalize_embeddings=True)
        return float(np.dot(emb1, emb2))

    # -------- save
    def save(self,retrieved_output, eval_summary, eval_metrices):
        if self.mode == AWS:
            bucket, key = s3_utills.get_s3_bucket_key(self.path)
            s3_client = boto3.client('s3')

            s3_client.put_object(
                Bucket=bucket,
                Key=f"{key}/summary.json",
                Body=json.dumps(eval_summary, indent=2)
            )
            s3_client.put_object(
                Bucket=bucket,
                Key=f"{key}/details.json",
                Body=json.dumps({'retrieved_output': retrieved_output, 'eval_metrices': eval_metrices}, indent=2)
            )
        else:
            os.makedirs(self.path, exist_ok=True)
            with open(f"{self.path}/summary.json", 'w') as f:
                json.dump(eval_summary, f, indent=2)

            with open(f"{self.path}/details.json", 'w') as f:
                json.dump({'retrieved_output': retrieved_output, 'eval_metrices': eval_metrices}, f, indent=2)


    def evaluate(self, k, retrieved_output):
        
        eval_metrices = {}

        total_recall = 0
        total_precision = 0
        total_mrr = 0
        total_sas = 0
        

        total_cost = 0
        total_latency = 0

        for i, output in enumerate(retrieved_output):
            chunk_id = output['chunk_id']
            retrieved_chunk_ids = output['retrieved_chunk_ids']
            retrieved_chunk_texts = output['retrieved_chunk_texts']
            qus = output['qus']
            context = output['context']
            generated_ans = output['generated_ans']
            ground_truth_ans = output['ground_truth_ans']
            k = output['k']
            cost = output['cost']
            latency = output['latency']

            # 1. Recall
            recall = self.compute_recall_single_chunk(chunk_id, retrieved_chunk_ids)

            # 2. Precision
            precision_at_k = self.compute_precision_at_k_single_chunk(chunk_id, retrieved_chunk_ids, k)

            # 3. MRR and SAS score
            mrr = self.compute_mrr(chunk_id, retrieved_chunk_ids)
            sas_score = self.compute_semantic_similarity(generated_ans, ground_truth_ans)

            eval_metrices[i] = self.EVAL_METRICES(k, recall, mrr, precision_at_k, sas_score, 0, 0, 0, cost, latency)

            print(" Qus : ", qus)
            print("Generated ans : ", generated_ans)
            print("Ground truth ans : ", ground_truth_ans)
            print("Matrices : recall, MRR, precision_at_k, sas_score : ", recall, mrr, precision_at_k, sas_score)
            print("Retrieval Cost : ", cost)
            print("Retrieval Latency : ", latency)

            total_recall += recall
            total_precision += precision_at_k
            total_mrr += mrr 
            total_sas += sas_score
            
            total_cost += cost
            total_latency += latency
        
        print("total cost : ", total_cost, ", total latency : ", total_latency)

        # API call
        # Answer faithfulness, relevancy and correctness 
        # 5. Use temp_eval_metrices for batch API processing

        answer_faithfulness = []
        answer_relevancy = []
        answer_llm_correctness = []

        if self.use_llm_judge:
            batch_size = 5
            for start in range(0, len(retrieved_output), batch_size):
                
                # Get batch
                batch = retrieved_output[start:start + batch_size]

                # Call to API with the batch 
                api_response = self.get_batch_llm_judgement(batch, start)

                judgement_text = api_response['response']
                batch_cost = api_response['cost'] 
                batch_latency = api_response['latency'] 

                # Update total
                total_cost += batch_cost
                total_latency += batch_latency

                #Parse response {chunk_id -> {faithfulness, relevancy, correctness}}
                batch_score = self.parse(judgement_text, batch)

                for local_idx, b in enumerate(batch):
                    i = start + local_idx
                    faithfulness, relevancy, correctness = batch_score[i]

                    # Update eval_matrices
                    #eval_metrices[chunk_id] = self.EVAL_METRICES(k, recall, mrr, precision_at_k, sas_score, 0, 0, 0, cost, latency)
                    em = eval_metrices[i]

                    eval_metrices[i] = self.EVAL_METRICES(em['k'], em['recall'], em['mrr'], em['precision_at_k'], em['sas_score'], 
                                                                faithfulness, relevancy, correctness,
                                                                em['cost'] + batch_cost/len(batch), em['latency'] + batch_latency/len(batch))
                    

                    answer_faithfulness.append(faithfulness)
                    answer_relevancy.append(relevancy)
                    answer_llm_correctness.append(correctness)

        # ------------ Summary 

        avg_ans_faithfulness = sum(answer_faithfulness)/len(answer_faithfulness) if self.use_llm_judge else 0
        avg_ans_relevancy = sum(answer_relevancy)/len(answer_relevancy) if self.use_llm_judge else 0
        avg_ans_lmm_correctness = sum(answer_llm_correctness) / len(answer_llm_correctness) if self.use_llm_judge else 0
        
        print("recall, avg ans faithfulness, avg ans releavncy, avg_ans_lmm_correctness : ", 
            total_recall / len(retrieved_output), avg_ans_faithfulness, avg_ans_relevancy, avg_ans_lmm_correctness)

        eval_summary = {
            'dataset_size': self.dataset_size,
            'k': k,
            'num_questions': len(retrieved_output),
            'recall': total_recall / len(retrieved_output),
            'precision': total_precision / len(retrieved_output),
            'mrr': total_mrr / len(retrieved_output),
            'sas': total_sas / len(retrieved_output),
            'avg_faithfulness': avg_ans_faithfulness,
            'avg_relevancy': avg_ans_relevancy,
            'avg_ans_lmm_correctness' : avg_ans_lmm_correctness,
            'total_cost': total_cost,
            'total_latency': total_latency,
        }

        self.save(retrieved_output, eval_summary, eval_metrices)

        print("\nEvaluation")
        print(retrieved_output[0]['retrieval_type'])
        print(retrieved_output[0]['chunk_type'])
        print("\nEval summary : ")

        print(eval_summary)

        return eval_summary


