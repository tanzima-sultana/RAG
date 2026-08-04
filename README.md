# RAG-Eval Pipeline

A retrieval-augmented generation (RAG) pipeline for benchmarking retrieval and answer quality. It compares three chunking strategies (fixed-size, sentence-aware, and semantic), three FAISS index types (FlatIP, IVF, HNSW), and three retrieval modes (dense, sparse, and hybrid). An optional cross-encoder reranking stage can also be applied before generation. Every configuration is scored on a fixed ground-truth set using both retrieval metrics (recall, precision, MRR (mean reciprocal rank), SAS (semantic answer similarity)) and RAGAS answer-quality metrics (faithfulness, relevancy, correctness). The full pipeline is also served through a FastAPI endpoint and load-tested under concurrent requests.

**Key Results:**

- Across datasets (5K, 20K, 50K) retrieval reached up to **0.94 recall@5** with precision between 0.40 and 0.58, both strongest under semantic chunking and hybrid retrieval.
- Faithfulness stayed at **0.98 on average** (never below 0.96) and relevancy at 0.93, while correctness averaged **0.82** — answers stayed grounded and on-topic even where they were not fully correct.
- Semantic chunking retrieved **~15% higher recall** than fixed-size and sentence-aware, but at roughly a third of their average chunk size (70 vs ~210 characters), which favors recall@5.
- HNSW came within **2% of the FlatIP exact-search baseline** (0.92 vs 0.94 recall@5), while IVF at default `nprobe` trailed by ~11% and was the slowest of the three.