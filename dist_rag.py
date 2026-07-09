import sys


from constants import DATA_PATH, DATASET_SIZE, FIXED_CHUNKING, SENTENCE_CHUNKING, SEMANTIC_CHUNKING
from load_dataset import load_parquet_dataset, get_sample_parquet
from spark_chunking import compute_chunks

# ----- main
if __name__ == "__main__":  
    print("Distribute RAG")

    # arg 1 : Dataset size
    dataset_size = int(sys.argv[1])

    # arg 2 : Chunk type
    chunk_type = sys.argv[2] if len(sys.argv) > 2 else FIXED_CHUNKING
    valid_chunk_types = [FIXED_CHUNKING, SENTENCE_CHUNKING, SEMANTIC_CHUNKING]
    if chunk_type not in valid_chunk_types:
        raise ValueError(f"chunk_type must be one of {valid_chunk_types}, got '{chunk_type}'")

    print("---------------------------")
    print("Dataset size : ", dataset_size, ", chunk_type : ", chunk_type)

    # ----- 1. Load dataset
    dataset = load_parquet_dataset()

    # ----- 2. Spark Chunking
    chunks = []
    avg_tokens = 0
    input_path = get_sample_parquet(dataset_size)
    chunk_path = compute_chunks(input_path, dataset_size, chunk_type)