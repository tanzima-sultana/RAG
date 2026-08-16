SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source ~/pyenv/bin/activate

export PYTHONPATH=$(pwd)
mkdir -p log

# ------------ Build

MODE="local" # local, aws
DEVICE="cuda"
MODEL_NAME="all-MiniLM-L6-v2"
DATASET_SIZE=20000
MAX_CHUNK_SIZE=256
FIX_CHUNK_OVERLAP=32
SEMANTIC_THRESHOLD=0.3
IVF_NLIST=256
HNSW_M=32

MOCK_RUN=0
NUM_QUERIES=50

# Set to 1 after changing chunking/embedding code or params and regenerating
# embeddings for a size/strategy that already has a Qdrant collection — the
# collection name doesn't change just because its contents did, so without
# this the old vectors would otherwise be silently left in place.
REBUILD_VECTORDB=0

BUILD_LOG="log/build_log.txt"

python3 scripts/build_rag.py \
  --mode $MODE \
  --device $DEVICE \
  --model_name $MODEL_NAME \
  --dataset_size $DATASET_SIZE \
  --max_chunk_size $MAX_CHUNK_SIZE \
  --fix_chunk_overlap $FIX_CHUNK_OVERLAP \
  --semantic_threshold $SEMANTIC_THRESHOLD \
  --ivf_nlist $IVF_NLIST \
  --hnsw_m $HNSW_M \
  --mock_run $MOCK_RUN \
  --num_queries $NUM_QUERIES \
  --rebuild_vectordb $REBUILD_VECTORDB \
  2>&1 | tee "$BUILD_LOG"

# ------------ Eval 


CHUNKING_TYPE="fixed" #"fixed", "sentence", "semantic"
INDEX_TYPE="hnsw" #"flatip", "ivf", "hnsw"
RETRIEVAL_TYPE="hybrid" #"dense", "bm25", "hybrid", "vectordb"
K=5
RE_RANKING=1 # 0 = no rerank, 1 = rerank
RERANK_K=20

EVAL_LOG="log/eval_log.txt"

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