#!/bin/bash

source ~/pyenv/bin/activate

MOCK_RUN=0
MODE="local"
DEVICE="cuda"
MODEL_NAME="all-MiniLM-L6-v2"
DATASET_SIZE=50000

CHUNKING_TYPE="fixed"
INDEX_TYPE="flatip"
RETRIEVAL_TYPE="hybrid"
K=5
RE_RANKING=1
RERANK_K=20

BASE_URL="http://127.0.0.1:8001"

CONCURRENCY_LEVELS="5 10"
REQUESTS_PER_LEVEL=1

python ~/Documents/AI/RAG/app/codes/concurrent_req.py \
  --base_url "$BASE_URL" \
  --mock_run $MOCK_RUN \
  --mode "$MODE" \
  --device "$DEVICE" \
  --model_name "$MODEL_NAME" \
  --dataset_size $DATASET_SIZE \
  --chunking_type "$CHUNKING_TYPE" \
  --index_type "$INDEX_TYPE" \
  --retrieval_type "$RETRIEVAL_TYPE" \
  --k $K \
  --re_ranking $RE_RANKING \
  --rerank_k $RERANK_K \
  --concurrency_levels $CONCURRENCY_LEVELS \
  --requests_per_level $REQUESTS_PER_LEVEL