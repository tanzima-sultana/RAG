source ~/pyenv/bin/activate

DATASET_SIZE=100

FIXED_CHUNKING="fixed"
SENTENCE_CHUNKING="sentence"
SEMANTIC_CHUNKING="semantic"

DENSE_RETRIEVAL="dense"
BM25_RETRIEVAL="bm25"
HYBRID_RETRIEVAL="hybrid"

RE_RANKING=1
TOP_K=20

k=5

export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export JAVA_TOOL_OPTIONS="--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED"
unset SPARK_HOME

python3 dist_rag.py $DATASET_SIZE $FIXED_CHUNKING "$@"