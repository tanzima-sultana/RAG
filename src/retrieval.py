from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import time
from src.anthropic_api import AnthropicAPI

# Eval Set format
#'chunk_id': chunk_id,
#'question': question,
#'answer': answer


class Retrieval:
   def __init__(self, dry_run, retrieval_type, chunks, eval_set, k, reranking, rerank_k, model_name, device):
      self.dry_run = dry_run
      self.retrieval_type = retrieval_type
      self.chunks = chunks 
      self.eval_set = eval_set 
      self.k = k 
      self.reranking = reranking
      self.rerank_k = rerank_k
      self.model_name = model_name
      self.device = device

      self.model = SentenceTransformer(self.model_name, device=self.device)
      self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') 
   
   def RETRIEVED_OUTPUT(self, chunk_id, retrieved_chunk_ids, retrieved_chunk_texts, qus, context, generated_ans, ground_truth_ans, k, cost, latency):
      return {
         'chunk_id' : chunk_id,
         'retrieved_chunk_ids' : retrieved_chunk_ids,
         'retrieved_chunk_texts' : retrieved_chunk_texts,
         'qus': qus,
         'context': context,
         'generated_ans': generated_ans,
         'ground_truth_ans': ground_truth_ans,
         'k' : k,
         'cost' : cost,
         'latency' : latency
      }

   def get_answers_batch(self, batch_items):

      if self.dry_run:
         response = ""
         for item in batch_items:
            response += f"\nID: {item['chunk_id']}\nAnswer: MOCK_ANSWER\n"

         return {'response': response, 'cost': 0, 'latency': 0}
   
      # batch_items: same as RETRIEVED_OUTPUT format
      blocks = ""
      for item in batch_items:
         blocks += f"\n[ID: {item['chunk_id']}]\nContext: {item['context']}\nQuestion: {item['qus']}\n"

      prompt = f"""You are a helpful assistant. For each numbered item below, use the given context to answer the question.
      If the answer is not contained within the context, say "I don't know."

      Respond with one block per item, in exactly this format, with no extra commentary:

      ID: <id>
      Answer: <answer>

      Items:
      {blocks}"""

      an = AnthropicAPI(anthropic_model="claude-sonnet-4-6", max_tokens=4000)
      api_response = an.anthropic_msg_api(prompt)
      return api_response

   def parse_answers_batch(self, response_text):
      results = {}
      current_id = None
      current_answer = None

      for line in response_text.strip().split('\n'):
         line = line.strip()
         if not line:
            continue
         if line.startswith('ID:'):
            if current_id is not None and current_answer is not None:
               results[current_id] = current_answer
            current_id = line.replace('ID:', '', 1).strip()
            current_answer = None
         elif line.startswith('Answer:'):
            current_answer = line.replace('Answer:', '', 1).strip()
         elif current_answer is not None:
            current_answer += ' ' + line

      if current_id is not None and current_answer is not None:
         results[current_id] = current_answer

      return results
   
   # 1. Dense

   def retrieval_dense(self, faiss_index):

      retrieved_outputs = []
      temp_outputs = []

      print("\n Dense Retrieval with reranking : ", self.reranking, ", and rerank_k : ", self.rerank_k)

      questions = [item['question'] for item in self.eval_set]

      # 1. Query embedding
      query_embeddings = self.model.encode(questions, normalize_embeddings=True)
      print("query_embeddings shape - ", query_embeddings.shape)

      # 2. Find the closest k (or rerank_k) indices
      if self.reranking == 1:
         distances, dense_indices = faiss_index.search(query_embeddings, self.rerank_k)
      else:
         distances, dense_indices = faiss_index.search(query_embeddings, self.k)
      

      chunk_id_to_text = {chunk['chunk_id']: chunk['chunk_text'] for chunk in self.chunks}

      for i, item in enumerate(self.eval_set):

         chunk_id = item['chunk_id']
         qus = item['question']
         ground_truth_ans = item['answer']

         # 3. Retrieve chunks_ids and chunk_texts
         retrieved_chunk_ids = []
         retrieved_chunk_texts = []

         search_indices = dense_indices[i]

         if self.reranking == 1:
            # already top rerank_k retrieved, now rerank down to k
            temp_retrieved_chunk_ids = []
            temp_retrieved_chunk_texts = []
            for j in search_indices:
               if j == -1:
                  continue 
               temp_retrieved_chunk_ids.append(self.chunks[j]['chunk_id'])
               temp_retrieved_chunk_texts.append(self.chunks[j]['chunk_text'])

            cross_encoder_scores = self.cross_encoder.predict([(qus, text) for text in temp_retrieved_chunk_texts])

            reranked = sorted(zip(temp_retrieved_chunk_ids, cross_encoder_scores), key=lambda x: x[1], reverse=True)

            retrieved_chunk_ids = [cid for cid, score in reranked][:self.k]
            retrieved_chunk_texts = [chunk_id_to_text[cid] for cid in retrieved_chunk_ids]
         else:
            for j in search_indices:
               if j == -1:
                  continue 
               retrieved_chunk_ids.append(self.chunks[j]['chunk_id'])
               retrieved_chunk_texts.append(self.chunks[j]['chunk_text'])
         
         # 4. Create context for each eval question
         context = ' '.join(retrieved_chunk_texts)
         temp_outputs.append(self.RETRIEVED_OUTPUT(chunk_id, retrieved_chunk_ids, retrieved_chunk_texts, qus, context, "", ground_truth_ans, self.k, "", ""))
      

      # 5. Use retrieved_outputs for batch API processing and update its empty field
      batch_size = 5
      for start in range(0, len(temp_outputs), batch_size):
         batch = temp_outputs[start:start + batch_size]

         s = time.time()
         api_response = self.get_answers_batch(batch)
         e = time.time() - s
         print("batch API response time : ", e) 
         
         answers = self.parse_answers_batch(api_response['response'])

         # For each ouput, fill out the empty field
         for out in temp_outputs[start:start + batch_size]:
            generated_ans = answers.get(out['chunk_id'], "MISSING")
            if generated_ans == "MISSING":
               print(f"WARNING: no answer returned for chunk_id {out['chunk_id']}")
            
            retrieved_outputs.append(self.RETRIEVED_OUTPUT(
               out['chunk_id'], out['retrieved_chunk_ids'], out['retrieved_chunk_texts'], out['qus'], out['context'],
               generated_ans, out['ground_truth_ans'], out['k'],
               api_response['cost'], api_response['latency']
            ))
      
      #print("Retrieved Output\n")
      #print(retrieved_outputs)

      return retrieved_outputs
   
   # -------------------- 2. BM25

   def tokenize_chunk_text(self, text):
        return text.lower().split()
   
   def retrieval_bm25(self, bm25_index):

      retrieved_outputs = []
      temp_outputs = []

      print("\n BM25 Retrieval with reranking : ", self.reranking, ", and rerank_k : ", self.rerank_k)

      questions = [item['question'] for item in self.eval_set]

      chunk_id_to_text = {chunk['chunk_id']: chunk['chunk_text'] for chunk in self.chunks}

      for i, item in enumerate(self.eval_set):

         chunk_id = item['chunk_id']
         qus = item['question']
         ground_truth_ans = item['answer']

         # Retrieve chunks_ids and chunk_texts
         retrieved_chunk_ids = []
         retrieved_chunk_texts = []

         tokenized_qus = self.tokenize_chunk_text(qus) # Tokenize qus
         scores = bm25_index.get_scores(tokenized_qus)  # one score per chunk, same order as chunks list

         if self.reranking == 1:
            search_indices = np.argsort(scores)[::-1][:self.rerank_k]  # top rerank_k
            # already top rerank_k retrieved, now rerank down to k
            temp_retrieved_chunk_ids = []
            temp_retrieved_chunk_texts = []
            for j in search_indices:
               if j == -1:
                  continue 
               temp_retrieved_chunk_ids.append(self.chunks[j]['chunk_id'])
               temp_retrieved_chunk_texts.append(self.chunks[j]['chunk_text'])

            cross_encoder_scores = self.cross_encoder.predict([(qus, text) for text in temp_retrieved_chunk_texts])

            reranked = sorted(zip(temp_retrieved_chunk_ids, cross_encoder_scores), key=lambda x: x[1], reverse=True)

            retrieved_chunk_ids = [cid for cid, score in reranked][:self.k]
            retrieved_chunk_texts = [chunk_id_to_text[cid] for cid in retrieved_chunk_ids]
         else:
            search_indices = np.argsort(scores)[::-1][:self.k]  #  take top k
            for j in search_indices:
               if j == -1:
                  continue 
               retrieved_chunk_ids.append(self.chunks[j]['chunk_id'])
               retrieved_chunk_texts.append(self.chunks[j]['chunk_text'])
         
         # 4. Create context for each eval question
         context = ' '.join(retrieved_chunk_texts)
         temp_outputs.append(self.RETRIEVED_OUTPUT(chunk_id, retrieved_chunk_ids, retrieved_chunk_texts, qus, context, "", ground_truth_ans, self.k, "", ""))

      # 5. Use retrieved_outputs for batch API processing and update its empty field
      batch_size = 5
      for start in range(0, len(temp_outputs), batch_size):
         batch = temp_outputs[start:start + batch_size]

         s = time.time()
         api_response = self.get_answers_batch(batch)
         e = time.time() - s
         print("batch API response time : ", e) 
         
         answers = self.parse_answers_batch(api_response['response'])

         # For each ouput, fill out the empty field
         for out in temp_outputs[start:start + batch_size]:
            generated_ans = answers.get(out['chunk_id'], "MISSING")
            if generated_ans == "MISSING":
               print(f"WARNING: no answer returned for chunk_id {out['chunk_id']}")
            
            retrieved_outputs.append(self.RETRIEVED_OUTPUT(
               out['chunk_id'], out['retrieved_chunk_ids'], out['retrieved_chunk_texts'], out['qus'], out['context'],
               generated_ans, out['ground_truth_ans'], out['k'],
               api_response['cost'], api_response['latency']
            ))
      
      #print("Bm25 -- Retrieved Output\n")
      #print(retrieved_outputs)

      return retrieved_outputs
   
   # -------------------- 3. Hybrid

   # ----- RRF : Reciprocal Rank Fushion ----- #

   def reciprocal_rank_fusion(self, dense_chunk_ids, bm25_chunk_ids, k_const=60):
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

   def retrieval_hybrid(self, faiss_index, bm25_index):

      if faiss_index == None or bm25_index == None:
         print("One of the index is empty. Error")
         return []

      retrieved_outputs = []
      temp_outputs = []

      print("\n Hybrid Retrieval with reranking : ", self.reranking, ", and rerank_k : ", self.rerank_k)

      questions = [item['question'] for item in self.eval_set]

      # 1. Query embedding
      query_embeddings = self.model.encode(questions, normalize_embeddings=True)
      print("query_embeddings shape - ", query_embeddings.shape)

      # 2. Find the closest k (or rerank_k) indices
      if self.reranking == 1:
         distances, dense_indices = faiss_index.search(query_embeddings, self.rerank_k)
      else:
         distances, dense_indices = faiss_index.search(query_embeddings, self.k)
      
      chunk_id_to_text = {chunk['chunk_id']: chunk['chunk_text'] for chunk in self.chunks}

      for i, item in enumerate(self.eval_set):

         chunk_id = item['chunk_id']
         qus = item['question']
         ground_truth_ans = item['answer']

         # 3. Retrieve chunks_ids and chunk_texts
         retrieved_chunk_ids = []
         retrieved_chunk_texts = []

         # ----- For dense 
         dense_search_indices = dense_indices[i]
         # Retieve chunks
         dense_retrieved_chunk_ids = []
         dense_retrieved_chunk_texts =[]
         for j in dense_search_indices:
               if j == -1:
                  continue 
               dense_retrieved_chunk_ids.append(self.chunks[j]['chunk_id'])
               dense_retrieved_chunk_texts.append(self.chunks[j]['chunk_text'])

         # ----- For BM25
         tokenized_qus = self.tokenize_chunk_text(qus) 
         scores = bm25_index.get_scores(tokenized_qus) 

         bm25_search_indices = []
         if self.reranking == 1:
               bm25_search_indices = np.argsort(scores)[::-1][:self.rerank_k]  
         else:
               bm25_search_indices = np.argsort(scores)[::-1][:self.k]  

         # Retieve chunks
         bm25_retrieved_chunk_ids = []
         bm25_retrieved_chunk_texts =[]
         for j in bm25_search_indices:
               if j == -1:
                  continue 
               bm25_retrieved_chunk_ids.append(self.chunks[j]['chunk_id'])
               bm25_retrieved_chunk_texts.append(self.chunks[j]['chunk_text'])
         
         # ----- re_ranking & K=20
         if self.reranking == 1:
               # Use rerank_k instaed of small k
               temp_retrieved_chunk_ids = self.reciprocal_rank_fusion(dense_retrieved_chunk_ids, bm25_retrieved_chunk_ids, k_const=60)[:self.rerank_k]
               temp_retrieved_chunk_texts = [chunk_id_to_text[cid] for cid in temp_retrieved_chunk_ids]

               cross_encoder_scores = self.cross_encoder.predict([(qus, text) for text in temp_retrieved_chunk_texts])
               # sort chunk_ids by score, descending, take top k
               reranked = sorted(zip(temp_retrieved_chunk_ids, cross_encoder_scores), key=lambda x: x[1], reverse=True)
               
               retrieved_chunk_ids = [cid for cid, score in reranked][:self.k]
               retrieved_chunk_texts = [chunk_id_to_text[cid] for cid in retrieved_chunk_ids]

         else:
               # RRF and slice at k
               retrieved_chunk_ids = self.reciprocal_rank_fusion(dense_retrieved_chunk_ids, bm25_retrieved_chunk_ids, k_const=60)[:self.k]
               retrieved_chunk_texts = [chunk_id_to_text[cid] for cid in retrieved_chunk_ids]
         
         # 4. Create context for each eval question
         context = ' '.join(retrieved_chunk_texts)
         temp_outputs.append(self.RETRIEVED_OUTPUT(chunk_id, retrieved_chunk_ids, retrieved_chunk_texts, qus, context, "", ground_truth_ans, self.k, "", ""))
      

      # 5. Use retrieved_outputs for batch API processing and update its empty field
      batch_size = 5
      for start in range(0, len(temp_outputs), batch_size):
         batch = temp_outputs[start:start + batch_size]

         s = time.time()
         api_response = self.get_answers_batch(batch)
         e = time.time() - s
         print("batch API response time : ", e) 
         
         answers = self.parse_answers_batch(api_response['response'])

         # For each ouput, fill out the empty field
         for out in temp_outputs[start:start + batch_size]:
            generated_ans = answers.get(out['chunk_id'], "MISSING")
            if generated_ans == "MISSING":
               print(f"WARNING: no answer returned for chunk_id {out['chunk_id']}")
            
            retrieved_outputs.append(self.RETRIEVED_OUTPUT(
               out['chunk_id'], out['retrieved_chunk_ids'], out['retrieved_chunk_texts'], out['qus'], out['context'],
               generated_ans, out['ground_truth_ans'], out['k'],
               api_response['cost'], api_response['latency']
            ))
      
      #print("Hybrid -- Retrieved Output\n")
      #print(retrieved_outputs)

      return retrieved_outputs


