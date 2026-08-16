#!/bin/bash

# Eval phase, split out of local_run.sh: retrieval + evaluation against the
# manifest produced by run/build.sh. MODE/DEVICE/MODEL_NAME/DATASET_SIZE/
# MOCK_RUN/NUM_QUERIES must match the build.sh run being evaluated.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source ~/pyenv/bin/activate

export PYTHONPATH=$(pwd)
mkdir -p log

MODE="local" # local, aws
DEVICE="cuda"
MODEL_NAME="all-MiniLM-L6-v2"
DATASET_SIZE=20000

MOCK_RUN=0
NUM_QUERIES=50

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
