
#!/bin/bash

MOCK_RUN=1
MODE="local" # local, aws
DEVICE="cuda"
MODEL_NAME="all-MiniLM-L6-v2"
DATASET_SIZE=50000

CHUNKING_TYPE="fixed" #"fixed", "sentence", "semantic"
INDEX_TYPE="flatip" #"flatip", "ivf", "hnsw"
RETRIEVAL_TYPE="hybrid" #"dense", "bm25", "hybrid", "vectordb"
NUM_QUERIES=50
K=5
RE_RANKING=1 # 0 = no rerank, 1 = rerank
RERANK_K=20

BASE_URL="http://127.0.0.1:8001"

curl -s "$BASE_URL/health"

curl -s -X POST "$BASE_URL/query" \
  -H "Content-Type: application/json" \
  -d "{
        \"mock_run\": $MOCK_RUN,
        \"mode\": \"$MODE\",
        \"device\": \"$DEVICE\",
        \"model_name\": \"$MODEL_NAME\",
        \"dataset_size\": $DATASET_SIZE,
        \"chunking_type\": \"$CHUNKING_TYPE\",
        \"index_type\": \"$INDEX_TYPE\",
        \"retrieval_type\": \"$RETRIEVAL_TYPE\",
        \"num_queries\": $NUM_QUERIES,
        \"k\": $K,
        \"re_ranking\": $RE_RANKING,
        \"rerank_k\": $RERANK_K
      }"