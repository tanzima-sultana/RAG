import os
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np

from config import ANTHROPIC_MSG_API_KEY
from constants import INPUT_COST_PER_MTOK, OUTPUT_COST_PER_MTOK
import random
import json
import time
import anthropic



class AnthropicAPI:
    def __init__(self, anthropic_model, max_tokens):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_MSG_API_KEY)
        self.anthropic_model = anthropic_model
        self.max_tokens = max_tokens

    def API_RESPONSE(self, response, input_tokens, output_tokens, cost, latency):

        return {
            'response': response,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost': cost,
            'latency': latency
        }


    def anthropic_msg_api(self, prompt):

        start = time.time()
        response = self.client.messages.create(
            model= self.anthropic_model, 
            max_tokens = self.max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        end = time.time()
        latency = end - start

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = (input_tokens / 1_000_000) * INPUT_COST_PER_MTOK + \
            (output_tokens / 1_000_000) * OUTPUT_COST_PER_MTOK

        response = response.content[0].text.strip()
        api_response = self.API_RESPONSE(response, input_tokens, output_tokens, cost, latency)
        
        return api_response

    