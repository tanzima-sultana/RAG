source ~/pyenv/bin/activate

DATASET_SIZE=100

FIXED="fixed"
SENTENCE="sentence"
SEMANTIC="semantic"

DENSE_RETRIEVAL="dense"
BM25_RETRIEVAL="bm25"
HYBRID_RETRIEVAL="hybrid"

k=5

python3 rag.py $DATASET_SIZE $FIXED $HYBRID_RETRIEVAL $k
