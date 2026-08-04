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

### Chunking

Semantic chunking led on recall (0.92) and precision (0.53), but at roughly a third of the chunk size of the other two strategies, which favors recall@5 and makes the gain misleading. Fixed and sentence-aware chunking were close to each other (recall 0.80 and 0.78), with fixed slightly ahead on ranking metrics.

### Retrieval

Hybrid retrieval gave the best balance, matching dense on recall (0.92) while improving MRR (0.88) and correctness (0.82) over both dense and BM25 alone. BM25 on its own had the weakest recall and relevancy, and dense on its own trailed hybrid on ranking quality, confirming that combining sparse and dense signals helped most.

### Reranking

Adding the cross-encoder reranker on top of hybrid retrieval raised precision (0.40 to 0.49) and MRR (0.81 to 0.86) while recall stayed flat at 0.88. This is the expected behavior of a reordering step: it promotes the relevant chunks already retrieved without changing which chunks were retrieved.

### Index type

At matched configuration, HNSW tracked the FlatIP exact-search baseline closely (0.92 vs 0.94 recall at 20K), while IVF at its default `nprobe` dropped to 0.84 — an under-probing effect, not a fundamental weakness. Qdrant at 50K held recall at 0.84, close to FlatIP hybrid at the same scale, confirming it as a viable managed alternative to the in-process FAISS indexes.

### Evaluation metrics across runs

Retrieval metrics spread the widest, reflecting the configuration differences: recall ranged 0.78 to 0.94 (mean 0.87), precision 0.39 to 0.58 (mean 0.49), MRR 0.71 to 0.91 (mean 0.84), and semantic answer similarity 0.66 to 0.80 (mean 0.74). The lower ends came from the weaker setups (BM25-only, IVF at default `nprobe`), and the upper ends from hybrid retrieval with reranking.

The LLM-as-judge answer-quality scores were higher and more stable. Faithfulness stayed between 0.96 and 0.99 (mean 0.98) across every run, showing answers remained grounded in retrieved context regardless of configuration. Relevancy ranged 0.84 to 0.98 (mean 0.93), dipping only where retrieval was weakest. Correctness was the lowest of the three at 0.72 to 0.89 (mean 0.82), tracking retrieval quality — when retrieval surfaced weaker context, answers stayed faithful and on-topic but were more often incomplete or wrong.

### How the carried-forward configuration was chosen

The benchmark was run in stages rather than as a full grid, to keep the number of runs and the API cost down (each run incurs answer generation and LLM-as-judge Claude API calls). Each stage fixed the winner of the previous one and varied only the next parameter.

First, chunking was compared. Semantic chunking scored the highest recall (0.92 vs 0.80 fixed, 0.78 sentence), but its average chunk size was far smaller (~70 vs ~210 characters of fixed and sentence), producing more and shorter chunks. Because recall@5 is mechanically easier to satisfy with smaller chunks, the comparison was not fair. So, fixed-size chunking was carried forward instead of semantic.

Retrieval was compared next, holding chunking fixed. Hybrid retrieval won over dense and BM25, so it was carried into the remaining stages. Reranking was then compared on top of hybrid; it improved ranking quality, so it was kept on. The winning configuration — fixed chunking, hybrid retrieval, reranking on — was then run across index types (FlatIP, IVF, HNSW, Qdrant) and dataset sizes (5K, 20K, 50K).

### Load Test

FastAPI `/query` endpoint, full pipeline, latency in milliseconds.

| Concurrency | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) | Error Rate |
| ----------- | -------- | -------- | -------- | ------------------ | ---------- |
| 5 | 2927 | 3299 | 3334 | 1.49 | 0% |
| 10 | 4777 | 6272 | 6714 | 1.46 | 0% |

The full serving path — query embedding, vector search, reranking, and answer generation — was load tested. Latency rose with concurrency: p50 went from 2.9s to 4.8s and p99 from 3.3s to 6.7s, while throughput held near 1.5 req/s and the error rate stayed at 0% across both runs. The latency is LLM-bound, driven by the answer-generation call rather than retrieval. The p50–p99 gap also widened at concurrency 10 (4.8s to 6.7s versus 2.9s to 3.3s at concurrency 5), the slowest requests stretching further as load increased.

### File sizes and generation times

File sizes in MB, generation times in seconds. Index time is the total build time across all indexes combined.

| Dataset | Strategy | Chunk Size | Chunk Time | Embed Size | Embed Time | FlatIP | IVF | HNSW | BM25 |
| ------- | -------- | ---------- | ---------- | ---------- | ---------- | ------ | ------ | ------ | ------ |
| 20K | Fixed | 72.91 | 33.91 | 108.77 | 99.06 | 105.47 | 106.40 | 124.16 | 82.48 |
| 20K | Sentence | 65.26 | 34.68 | 104.51 | 97.78 | 101.42 | 102.32 | 119.39 | 77.91 |
| 20K | Semantic | 71.49 | 320.28 | 299.72 | 120.36 | 290.78 | 292.67 | 342.03 | 84.72 |
| 50K | Fixed | 183.72 | 90.75 | 273.92 | 242.90 | 265.80 | 267.56 | 312.90 | 204.16 |
| 50K | Sentence | 164.46 | 90.85 | 263.38 | 240.86 | 255.57 | 257.28 | 300.85 | 191.63 |
| 50K | Semantic | 180.16 | 791.36 | 755.30 | 297.73 | 732.78 | 736.97 | 862.60 | 208.71 |

Total index build time (all indexes combined): 31.18s at 20K, 129.67s at 50K.