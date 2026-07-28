source ~/pyenv/bin/activate

export PYTHONPATH=$(pwd)

MODE="local" # local, aws
DEVICE="cuda"
MODEL_NAME="all-MiniLM-L6-v2"
DATASET_SIZE=1000
CHUNKING_TYPE="fixed" #"fixed", "sentence", "semantic"
MAX_CHUNK_SISZE=256
FIX_CHUNK_OVERLAP=32
SEMANTIC_THREASHOLD=0.3
INDEXING_TYPE="flatip" #flatip, ivf, hnsw
NO_EVAL_QUERY=20
K=5
RETRIEVAL_TYPE="dense" #"dense", "bm25", "hybrid"
RE_RANKING=1 # 0 = no rerank, 1 = rerank
RERANK_K=20

OUTPUT_FILE="output.txt"

python3 scripts/run_rag.py \
  --mode $MODE \
  --device $DEVICE \
  --model_name $MODEL_NAME \
  --dataset_size $DATASET_SIZE \
  --chunking_type $CHUNKING_TYPE \
  --max_chunk_size $MAX_CHUNK_SISZE \
  --fix_chunk_overlap $FIX_CHUNK_OVERLAP \
  --semantic_threshold $SEMANTIC_THREASHOLD \
  --indexing_type $INDEXING_TYPE \
  --no_eval_query $NO_EVAL_QUERY \
  --k $K \
  --retrieval_type $RETRIEVAL_TYPE \
  --reranking $RE_RANKING\
  --rerank_k $RERANK_K \
  2>&1 | tee "$OUTPUT_FILE"