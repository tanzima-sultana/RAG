# RAG-Eval Pipeline

A retrieval-augmented generation (RAG) pipeline for benchmarking retrieval and answer quality. It compares three chunking strategies (fixed-size, sentence-aware, and semantic), three FAISS index types (FlatIP, IVF, HNSW), and three retrieval modes (dense, sparse, and hybrid). An optional cross-encoder reranking stage can also be applied before generation. Every configuration is scored on a fixed ground-truth set using both retrieval metrics (recall, precision, MRR, SAS (semantic answer similarity)) and LLM-as-judge answer-quality metrics (faithfulness, relevancy, correctness). The full pipeline is also served through a FastAPI endpoint and load-tested under concurrent requests.

**Key Results:**

- Across datasets (5K, 20K, 50K) retrieval reached up to **0.94 recall@5** with precision between 0.40 and 0.58, both strongest under semantic chunking and hybrid retrieval.
- Faithfulness stayed at **0.98 on average** (never below 0.96) and relevancy at 0.93, while correctness averaged **0.82** — answers stayed grounded and on-topic even where they were not fully correct.
- Semantic chunking retrieved **~15% higher recall** than fixed-size and sentence-aware, but at roughly a third of their average chunk size (70 vs ~210 characters), which favors recall@5.
- HNSW came within **2% of the FlatIP exact-search baseline** (0.92 vs 0.94 recall@5), while IVF at default `nprobe` trailed by ~11% and was the slowest of the three.

## Architecture

The pipeline runs in two phases: an offline build phase that generates all chunking, embedding, index, and eval-questions up front, and another pahse (retrieval+eval or FastApi service) phase that reads those saved files to answer queries and evaluate configurations.

### Build Phase 

Run once per dataset size

```
                              Raw Corpus
                                  |
         +------------------------+------------------------+
         |                        |                        |
       Fixed              Sentence-aware               Semantic        
         |                        |                        |
   +-----+-----+            +-----+-----+            +-----+-----+
   |     |     |            |     |     |            |     |     |
 Embed BM25 EvalQs        Embed BM25 EvalQs        Embed BM25 EvalQs     
   |                        |                        |
   +--------+               +--------+               +--------+
   |        |               |        |               |        |
 FAISS    Qdrant            FAISS  Qdrant            FAISS   Qdrant           
   |                        |                        |
   +------------------------+------------------------+
                            |
                            v
             Save on disk : chunks, embeddings, indexes, eval sets

FAISS = FlatIP, IVF, HNSW
```

### Retrieval + Eval phase 

```
   Query 
     |
     v
  Embed query
     |
     v
  Retrieve  -->  dense (FAISS) | sparse (BM25) | hybrid (RRF fusion)
     |
     v
  Rerank (optional)  -->  cross-encoder reorders top-k
     |
     v
  Generate  -->  Claude API, answer grounded in retrieved chunks
     |
     v
  Response: answer + source chunks 
     |
     v
  Evaluation (recall, precision, mrr ...........)

```

The same path backs two entry points: one single evaluation for each run, which evaluates teh RAG pipeline across the configuration and scores the output against the ground-truth eval set, and another one is FastAPI service, which exposes it as a `/query` endpoint for live requests and load testing.

## Tech Stack

### Retrieval & Embeddings
- sentence-transformers (`all-MiniLM-L6-v2`)
- FAISS — vector similarity search:
  - FlatIP — Flat index with Inner Product (exact, brute-force search)
  - IVF — Inverted File index (clusters vectors, searches nearest partitions)
  - HNSW — Hierarchical Navigable Small World (graph-based approximate search)
- Qdrant (via Docker) — vector database backend, an alternative to FAISS 
- BM25 — Best Matching 25, a sparse lexical ranking function
- Reciprocal Rank Fusion (RRF) — combines dense and bm25 rankings for hybrid retrieval
- Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) — reorders top-k candidates before generation

### Generation & Evaluation
- Anthropic Claude API for answer generation 
- LLM-as-judge scoring (faithfulness, relevancy, correctness)
- retrieval metrics 
    - recall
    - precision
    - MRR(Mean Reciprocal Rank)
    - semantic answer similarity

### Serving
- FastAPI + Uvicorn — `/query` serving layer
- httpx / asyncio — concurrent load testing

### Core
- Python
- NumPy
- PyTorch 

## Performance Summary

| Dataset | Chunking | Index | Retrieval | Rerank | Recall | Precision | MRR | SAS | Faithfulness | Relevancy | Correctness | Cost ($) |
| ------- | -------- | ------- | --------- | ------ | ------ | --------- | ---- | ---- | ------------ | --------- | ----------- | -------- |
| 50K | fixed | FlatIP | dense | no | 0.80 | 0.41 | 0.75 | 0.74 | 0.96 | 0.92 | 0.74 | 1.30 |
| 50K | sentence | FlatIP | dense | no | 0.78 | 0.42 | 0.71 | 0.66 | 0.99 | 0.91 | 0.72 | 1.34 |
| 50K | semantic | FlatIP | dense | no | 0.92 | 0.53 | 0.86 | 0.73 | 0.97 | 0.92 | 0.74 | 0.80 |
| 50K | semantic | FlatIP | BM25 | no | 0.84 | 0.39 | 0.81 | 0.68 | 0.99 | 0.84 | 0.76 | 0.78 |
| 50K | semantic | FlatIP | hybrid | no | 0.92 | 0.51 | 0.88 | 0.78 | 0.96 | 0.94 | 0.82 | 0.82 |
| 50K | fixed | FlatIP | hybrid | no | 0.88 | 0.40 | 0.81 | 0.80 | 0.96 | 0.95 | 0.85 | 1.35 |
| 50K | fixed | FlatIP | hybrid | yes | 0.88 | 0.49 | 0.86 | 0.80 | 0.98 | 0.93 | 0.85 | 1.45 |
| 50K | fixed | Qdrant | vectordb | yes | 0.84 | 0.45 | 0.83 | 0.78 | 0.98 | 0.93 | 0.81 | 1.44 |
| 5K | fixed | FlatIP | hybrid | yes | 0.90 | 0.58 | 0.90 | 0.74 | 0.99 | 0.96 | 0.89 | 1.42 |
| 5K | fixed | IVF | hybrid | yes | 0.88 | 0.54 | 0.88 | 0.71 | 0.98 | 0.97 | 0.88 | 1.42 |
| 5K | fixed | HNSW | hybrid | yes | 0.90 | 0.58 | 0.90 | 0.76 | 0.98 | 0.98 | 0.89 | 1.41 |
| 20K | fixed | FlatIP | hybrid | yes | 0.94 | 0.54 | 0.91 | 0.74 | 0.98 | 0.97 | 0.88 | 1.45 |
| 20K | fixed | IVF | hybrid | yes | 0.84 | 0.45 | 0.80 | 0.67 | 0.99 | 0.88 | 0.79 | 1.43 |
| 20K | fixed | HNSW | hybrid | yes | 0.92 | 0.53 | 0.89 | 0.72 | 0.98 | 0.94 | 0.85 | 1.43 |