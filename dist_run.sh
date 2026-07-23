source ~/pyenv/bin/activate

export PYTHONPATH=$(pwd)

MODE="local" # local, aws
DEVICE="cpu" # cpu, cuda # for local, its cpu and gpu, for aws, its spark(cpu) and spark-gpu

DATASET_SIZE=100
CHUNKING_TYPE="fixed" #"fixed", "sentence", "semantic"
MAX_CHUNK_SISZE=256
FIX_CHUNK_OVERLAP=32
SEMANTIC_THREASHOLD=0.3
K=5
RETRIEVAL_TYPE="dense" #"dense", "bm25", "hybrid"
RE_RANKING=1 # 0 = no rerank, 1 = rerank
RERANK_K=20


python3 scripts/run_rag.py \
  --mode $MODE \
  --device $DEVICE \
  --dataset_size $DATASET_SIZE \
  --chunking_type $CHUNKING_TYPE \
  --max_chunk_size $MAX_CHUNK_SISZE \
  --fix_chunk_overlap $FIX_CHUNK_OVERLAP \
  --semantic_threshold $SEMANTIC_THREASHOLD \
  --k $K \
  --retrieval_type $RETRIEVAL_TYPE \
  --reranking $RE_RANKING\
  --rerank_k $RERANK_K