import random
import os
import json
import io
import boto3
import pickle

from config import S3_BUCKET
from constants import SEED, LOCAL, AWS
from src.dist import s3_utills
from src.anthropic_api import AnthropicAPI

class EvalQA:
   def __init__(self, mock_run, mode, dataset_size, num_queries):
      self.mock_run = mock_run
      self.mode = mode
      self.dataset_size = dataset_size
      self.num_queries = num_queries

      self.path = f"eval_qa/{dataset_size}_{num_queries}"
      if self.mode == AWS:
            self.path = f"s3://{S3_BUCKET}/" + self.path

   def EVAL_QA(self, chunk_id, question, answer):

      return {
         'chunk_id': chunk_id,
         'question': question,
         'answer': answer
      }
   
   def get_sample_chunks(self, chunks, min_chunk_size, no_of_chunks):
      temp_chunks = [c for c in chunks if c['chunk_size'] >= min_chunk_size]
      if self.num_queries > len(temp_chunks):
         raise ValueError(f"Requested {self.num_queries} samples but only {len(temp_chunks)} chunks meet min_chunk_size={min_chunk_size}")
      random.seed(SEED)
      return random.sample(temp_chunks, no_of_chunks)
   
   def parse_qa_batch_response(self, response_text):
      results = {}
      current = {}

      for line in response_text.strip().split('\n'):
         line = line.strip()
         if not line:
               continue
         if line.startswith('CHUNK_ID:'):
               if current.get('chunk_id') and current.get('question') and current.get('answer'):
                  results[current['chunk_id']] = {'question': current['question'], 'answer': current['answer']}
               current = {'chunk_id': line.replace('CHUNK_ID:', '', 1).strip()}
         elif line.startswith('Question:'):
               current['question'] = line.replace('Question:', '', 1).strip()
         elif line.startswith('Answer:'):
               current['answer'] = line.replace('Answer:', '', 1).strip()
         elif 'answer' in current:
               current['answer'] += ' ' + line

      if current.get('chunk_id') and current.get('question') and current.get('answer'):
         results[current['chunk_id']] = {'question': current['question'], 'answer': current['answer']}

      return results
   
   def generate_qa_batch(self, chunks):
      if self.mock_run == 1:
         mock_response = ""
         for chunk in chunks:
            mock_response += f"CHUNK_ID: {chunk['chunk_id']}\nQuestion: Mock question for {chunk['chunk_id']}?\nAnswer: Mock answer for {chunk['chunk_id']}.\n\n"
         return mock_response
   
      passages_block = ""
      for chunk in chunks:
         passages_block += f"\n[CHUNK_ID: {chunk['chunk_id']}]\n{chunk['chunk_text']}\n"

      prompt = f"""Below are {len(chunks)} passages, each labeled with a CHUNK_ID.

      For EACH passage, write one factual question whose answer is directly contained in that passage.
      The question should be phrased naturally, the way a real user would ask it — not by echoing the passage's exact wording.
      Then give the answer, using the passage's information.

      Respond with one block per passage, in exactly this format, with no extra commentary:

      CHUNK_ID: <chunk_id>
      Question: <question>
      Answer: <answer>

      Passages:
      {passages_block}"""

      an = AnthropicAPI(anthropic_model="claude-sonnet-4-6", max_tokens=4000)
      api_response = an.anthropic_msg_api(prompt)
      return api_response['response']
   

   def is_exists(self,eval_path):
        if self.mode == AWS:
            return s3_utills.s3_file_exists(eval_path)
        else:
            return os.path.exists(eval_path)
   
   def get_eval_set(self, eval_path):
      if self.is_exists(eval_path):
         if self.mode == AWS:
            bucket, key = s3_utills.get_s3_bucket_key(eval_path)
            buffer = io.BytesIO()
            boto3.client('s3').download_fileobj(bucket, key, buffer)
            buffer.seek(0)
            print("Loading eval_qa from disk")
            eval_set = json.load(buffer)
            if len(eval_set) == self.num_queries:
               return eval_set
         else:
            with open(eval_path, 'r') as f:
                  print("Loading eval_qa from disk")
                  eval_set = json.load(f)
            if len(eval_set) == self.num_queries:
                  return eval_set
      return None
       
   def build_eval_set(self, chunking_type, chunks_path, min_chunk_size):
      
      eval_path = self.path + f"_{chunking_type}"

      # Avoid mock
      if self.mock_run == 1:
         eval_path = eval_path + f"_mock"

      eval_set = self.get_eval_set(eval_path)
      if eval_set:
         return eval_path

      # Load chunks
      chunks_map = None
      with open(chunks_path, "rb") as f:
            chunks_map = pickle.load(f)
            
      chunk_ids = chunks_map.keys()
      chunks = chunks_map.values()
          
      print("Build eval_set")
      eval_set = []

      sample_chunks = self.get_sample_chunks(chunks, min_chunk_size, self.num_queries)
      
      raw_response = self.generate_qa_batch(sample_chunks)
      # parsed - {chunk_id -> (q, a)}
      parsed = self.parse_qa_batch_response(raw_response)
      # chunk_id and chunk map
      chunks_by_id = {c['chunk_id']: c for c in sample_chunks}

      for chunk_id, qa in parsed.items():
         chunk = chunks_by_id.get(chunk_id)
         if chunk is None:
               print(f"WARNING: model returned unknown chunk_id {chunk_id}, skipping")
               continue
         eval_set.append(self.EVAL_QA(chunk_id, qa['question'], qa['answer']))

      missing = set(chunks_by_id.keys()) - set(parsed.keys())
      if missing:
         print(f"WARNING: {len(missing)} chunks got no Q/A from model: {missing}")

      os.makedirs(os.path.dirname(eval_path), exist_ok=True)
      with open(eval_path, 'w') as f:
         json.dump(eval_set, f, indent=2)

      return eval_path