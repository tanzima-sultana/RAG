import glob
DATASET = glob.glob(
        "/home/tanzima/.cache/huggingface/hub/datasets--wikimedia--wikipedia/snapshots/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/20231101.en/*.parquet"
    )
DATASET_SIZE = 50000
DATA_PATH = "data/processed_dataset"
SEED = 42

# Chunking type
FIXED_CHUNKING="fixed"
SENTENCE_CHUNKING="sentence"
SEMANTIC_CHUNKING="semantic"

MAX_CHUNK_SIZE = 256
FIXED_CHUNK_OVERLAP = 64
SEMANTIC_THREASHOLD = 0.2