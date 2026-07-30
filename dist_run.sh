source ~/pyenv/bin/activate

export PYTHONPATH="$(pwd):$PYTHONPATH"

export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export JAVA_TOOL_OPTIONS="--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED"
unset SPARK_HOME

MODE="local" # local, aws
NUM_PARTITION=4
DEVICE="cuda" # cpu, cuda # for aws, its spark(cpu) and spark-gpu
MODEL_NAME="all-MiniLM-L6-v2"
DATASET_SIZE=1000
BATCH_SIZE=256
CHUNKING_TYPE="fixed" #"fixed", "sentence", "semantic"
MAX_CHUNK_SISZE=256
FIX_CHUNK_OVERLAP=32
SEMANTIC_THREASHOLD=0.3
INDEXING_TYPE="flatip" #flatip, ivf, hnsw

OUTPUT_FILE="log.txt"

python3 scripts/run_dist_rag.py \
  --mode $MODE \
  --num_partition $NUM_PARTITION \
  --device $DEVICE \
  --model_name $MODEL_NAME \
  --dataset_size $DATASET_SIZE \
  --batch_size $BATCH_SIZE \
  --chunking_type $CHUNKING_TYPE \
  --max_chunk_size $MAX_CHUNK_SISZE \
  --fix_chunk_overlap $FIX_CHUNK_OVERLAP \
  --semantic_threshold $SEMANTIC_THREASHOLD \
  --indexing_type $INDEXING_TYPE \
  2>&1 | tee "$OUTPUT_FILE"