#!/bin/bash

# Build phase, split out of local_run.sh: chunking, embedding, indexing,
# vector DB, and eval-question generation. Run run/eval.sh separately
# afterward for retrieval + evaluation.

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
  2>&1 | tee "$BUILD_LOG"
