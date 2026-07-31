source ~/pyenv/bin/activate

export PYTHONPATH=$(pwd)

# ------------ Build 

MODE="local" # local, aws
DEVICE="cuda"
MODEL_NAME="all-MiniLM-L6-v2"
DATASET_SIZE=1000
MAX_CHUNK_SISZE=256
FIX_CHUNK_OVERLAP=32
SEMANTIC_THREASHOLD=0.3
IVF_NLIST=256
HNSW_M=32

BUILD_LOG="build_log.txt"

python3 scripts/build_rag.py \
  --mode $MODE \
  --device $DEVICE \
  --model_name $MODEL_NAME \
  --dataset_size $DATASET_SIZE \
  --max_chunk_size $MAX_CHUNK_SISZE \
  --fix_chunk_overlap $FIX_CHUNK_OVERLAP \
  --semantic_threshold $SEMANTIC_THREASHOLD \
  --ivf_nlist $IVF_NLIST \
  --hnsw_m $HNSW_M \
  2>&1 | tee "$BUILD_LOG"

# ------------ Eval 

MOCK_RUN=1
CHUNKING_TYPE="fixed" #"fixed", "sentence", "semantic"
INDEX_TYPE="flatip" #"flatip", "ivf", "hnsw"
RETRIEVAL_TYPE="dense" #"dense", "bm25", "hybrid", "vectordb"
NUM_QUERIES=20
K=5
RE_RANKING=0 # 0 = no rerank, 1 = rerank
RERANK_K=20

EVAL_LOG="eval_log.txt"

python3 scripts/eval_rag.py \
  --mock_run $MOCK_RUN \
  --mode $MODE \
  --device $DEVICE \
  --model_name $MODEL_NAME \
  --dataset_size $DATASET_SIZE \
  --chunking_type $CHUNKING_TYPE \
  --index_type $INDEX_TYPE \
  --retrieval_type $RETRIEVAL_TYPE \
  --num_queries $NUM_QUERIES \
  --k $K \
  --re_ranking $RE_RANKING \
  --rerank_k $RERANK_K \
  2>&1 | tee "$EVAL_LOG"