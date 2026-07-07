from constants import INPUT_COST_PER_MTOK, OUTPUT_COST_PER_MTOK

def estimate_cost(eval_set_size, k, avg_tokens_per_chunk,
                   use_faithfulness, use_relevancy, use_llm_correctness,
                   avg_output_tokens=150):

    context_tokens = k * avg_tokens_per_chunk
    prompt_overhead = 100  # question text + instructions, rough

    calls_per_question = 1 + int(use_faithfulness) + int(use_relevancy) + int(use_llm_correctness)

    input_tokens_per_call = context_tokens + prompt_overhead
    cost_per_call = (input_tokens_per_call / 1_000_000) * INPUT_COST_PER_MTOK + \
                    (avg_output_tokens / 1_000_000) * OUTPUT_COST_PER_MTOK

    total_cost = eval_set_size * calls_per_question * cost_per_call

    return total_cost

# ----- main
if __name__ == "__main__":

    eval_set_size = 50 
    k = 5
    avg_tokens_per_chunk = 200
    use_faithfulness = True 
    use_relevancy = True 
    use_llm_correctness  = True 
    cost = estimate_cost(eval_set_size, k, avg_tokens_per_chunk,
                   use_faithfulness, use_relevancy, use_llm_correctness,
                   avg_output_tokens=150)
    
    print(cost)
    