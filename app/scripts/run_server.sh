#!/bin/bash

source ~/pyenv/bin/activate

export PYTHONPATH=$(pwd)

MODE="local" # local, aws
DATASET_SIZE=5000
CHUNKING_TYPE="fixed" #"fixed", "sentence", "semantic"
INDEX_TYPE="flatip" #"flatip", "ivf", "hnsw"

export mode=$MODE
export dataset_size=$DATASET_SIZE
export chunking_type=$CHUNKING_TYPE
export index_type=$INDEX_TYPE

PORT=8001
fuser -k $PORT/tcp || true
uvicorn app.codes.fast_api:app --reload --host 0.0.0.0 --port $PORT