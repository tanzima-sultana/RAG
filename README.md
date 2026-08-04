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

Semantic chunking led on recall (0.92) and precision (0.53). But its chunks were about a third the size of the other two strategies. Smaller chunks make recall@5 easier to hit, so the higher score is misleading rather than a real gain. Fixed and sentence-aware chunking were close to each other (recall 0.80 and 0.78), with fixed slightly ahead on ranking metrics. The semantic chunking used a similarity threshold of 0.3. But the threshold is already low, yet the chunks size still came out small. Lowering it further would merge unrelated sentences and blur the meaning of each chunk, so it was left at 0.3.

### Index type

HNSW stayed close to the FlatIP exact-search baseline (0.92 vs 0.94 recall at 20K). IVF dropped to 0.84. The reason is that `nprobe` was left at its default of 1, so each query searched only one of the 256 partitions. That is too little coverage — the drop is an under-probing effect, not a weakness in IVF itself. IVF was built with `nlist=256` and HNSW with `M=32`. A fuller sweep of `nprobe` and `M` is left for later benchmarking.

### Retrieval

Hybrid retrieval gave the best balance. It matched dense on recall (0.92) and beat both dense and BM25 on MRR (0.88) and correctness (0.82). BM25 alone had the weakest recall and relevancy. Dense alone trailed hybrid on ranking quality. Combining the sparse and dense signals is what helped most.

### Reranking

Adding the cross-encoder reranker on top of hybrid retrieval raised precision (0.40 to 0.49) and MRR (0.81 to 0.86) while recall stayed flat at 0.88. This is the expected behavior of a reordering step: it promotes the relevant chunks already retrieved without changing which chunks were retrieved.

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

Fixed and sentence-aware chunking have identical across file sizes and build times. Semantic chunking is the outlier: because it produces many more, smaller chunks, its chunking step runs about 9x slower than fixed at 50K, and its embedding and index files are roughly 2–3x larger. Total index build time stays low — around 2 minutes at 50K — so indexing is never the bottleneck.

At 50K, the full artifact set totals about 6.6 GB: chunks 528 MB, embeddings 1.29 GB, FAISS indexes 3.99 GB (FlatIP, IVF, HNSW across all three chunking strategies), BM25 605 MB, and Qdrant 376 MB.

## Evaluation Methodology

### Eval set

Eval questions are generated by Claude. A separate eval set is built for each dataset size and each chunking strategy. For every question, Claude is given a sampled chunk and produces a question and a reference answer from that chunk's text. The source document of that chunk is recorded as the ground truth for retrieval scoring.

Because each chunking strategy produces different chunk boundaries, the eval set is regenerated per strategy rather than shared. This keeps every question grounded in a real chunk of that strategy, but it means the three per-strategy sets are not directly comparable to each other — a question written from a semantic chunk is not the same question written from a fixed chunk. Comparisons within a strategy (across index type, retrieval mode, dataset size) are controlled; the cross-strategy chunking comparison is not.

### Retrieval metrics

Recall, precision, and MRR are scored by document match, not chunk-text match. Each chunk ID has the form `doc_id_chunk_id`, and the source `doc_id` is extracted from it. A retrieved chunk counts as a hit when its `doc_id` matches the ground-truth document. This is stable across dataset sizes, where individual chunk IDs are not.

### Answer-quality metrics

Faithfulness, relevancy, and correctness are scored by the Claude Messages API as an LLM-as-judge. For each answer, Claude returns a score from 0 to 1 on each of the three dimensions, given the question, the generated answer, the retrieved context, and the reference answer. The judge calls are batched to reduce API cost. The final score is the average across the entire eval set.

Semantic answer similarity (SAS) is computed without the judge: it is the cosine similarity between the embedding of the generated answer and the embedding of the reference answer.

## FastAPI + Load Testing

The pipeline is served through a FastAPI app. The retrieval configuration (retriever mode, top_k, rerank on/off) is set per request, and the models are loaded once at startup rather than per request.

- `fast_api.py` — FastAPI app; defines the `/query` endpoint
- `state.py` — loads models (SentenceTransformer, cross-encoder) once at startup and holds shared state
- `schema.py` — request/response Pydantic models
- `query.py` — query logic: embed → retrieve → rerank → generate
- `concurrent_req.py` — load-test client (async httpx, concurrency sweep)
- `run_server.sh` — starts the server (uvicorn); config via env vars, port cleanup
- `load_test.sh` — runs the load test

## Repository Structure

```
RAG/
├── app/
│   ├── codes/
│   │   ├── fast_api.py          # FastAPI app; /query endpoint
│   │   ├── state.py             # loads models once at startup, holds shared state
│   │   ├── schemas.py           # request/response Pydantic models
│   │   ├── query.py             # query logic: embed → retrieve → rerank → generate
│   │   └── concurrent_req.py    # load-test client (async httpx)
│   └── scripts/
│       ├── run_server.sh        # start the server (uvicorn)
│       ├── load_test.sh         # run the load test
│       └── query_curl.sh        # sample curl request
├── scripts/
│   ├── build_rag.py             # build phase: chunk, embed, index, eval sets
│   ├── eval_rag.py              # retrieval + evaluation
│   └── run_dist_rag.py          # distributed run entry point
├── src/
│   ├── dist/
│   │   ├── chunking_embedding.py
│   │   └── s3_utills.py
│   ├── local/
│   │   ├── chunking.py          # fixed / sentence / semantic chunking
│   │   └── embedding.py         # embedding generation
│   ├── dataset.py               # dataset load
│   ├── indexing.py              # FAISS FlatIP / IVF / HNSW
│   ├── vector_db.py             # Qdrant backend
│   ├── retrieval.py             # dense / sparse / hybrid retrieval
│   ├── eval_qa.py               # eval question generation
│   ├── evaluation.py            # metrics + LLM-as-judge
│   └── anthropic_api.py         # Claude API wrapper
├── config.py.template
├── constants.py
├── local_run.sh
├── dist_run.sh
├── qdrant_setup.md
└── README.md
```

## Installation / Setup

### Prerequisites

- Python 3.9
- Docker (runs the Qdrant vector database)
- An Anthropic API key (for answer generation and LLM-as-judge evaluation)

### Setup

**1. Clone the repository**

```bash
git clone https://github.com/tanzima-sultana/rag-eval-pipeline.git
cd rag-eval-pipeline
```

**2. Install requirements**

```bash
pip install -r requirements.txt
```

### Setting up Qdrant

Qdrant runs in Docker. Full step-by-step instructions (installing Docker Engine, persistent storage, verification) are in [`qdrant_setup.md`](qdrant_setup.md). The short version:

**1. Run Qdrant**

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v <qdrant-dir>/qdrant:/qdrant/storage qdrant/qdrant
```

Port 6333 is the REST API, 6334 is gRPC, and the `-v` mount persists data across container restarts.

**2. Verify it's up**

```bash
curl http://localhost:6333
```

**3. Install the Python client**

```bash
pip install qdrant-client
```
## How to Run

**1. Start Qdrant**

Make sure Docker is running, then start the Qdrant container (see [Setting up Qdrant](#setting-up-qdrant)).

**2. Local run (build + eval)**

```bash
./local_run.sh
```

This runs `build_rag.py` (chunking, embedding, indexing, eval-set generation) followed by `eval_rag.py` (retrieval and evaluation).

**3. Load test**

In a separate terminal, from `app/scripts`:

```bash
./run_server.sh     # starts the FastAPI server
./load_test.sh      # runs the load test against it
```

### Configuration

Run parameters are set as variables at the top of `local_run.sh`. Edit them before running.

**Build**

- `MODE` — `local` or `aws`
- `DEVICE` — `cuda` or `cpu`
- `MODEL_NAME` — embedding model (`all-MiniLM-L6-v2`)
- `DATASET_SIZE` — number of documents
- `MAX_CHUNK_SIZE` — max chunk size
- `FIX_CHUNK_OVERLAP` — overlap for fixed-size chunking
- `SEMANTIC_THRESHOLD` — similarity threshold for semantic chunking
- `IVF_NLIST` — number of IVF partitions
- `HNSW_M` — neighbors linked per node in HNSW
- `MOCK_RUN` — `0` for real API calls, `1` for mock responses (test pipeline without cost)
- `NUM_QUERIES` — number of eval questions

**Eval**

- `CHUNKING_TYPE` — `fixed`, `sentence`, or `semantic`
- `INDEX_TYPE` — `flatip`, `ivf`, or `hnsw`
- `RETRIEVAL_TYPE` — `dense`, `bm25`, `hybrid`, or `vectordb`
- `K` — number of chunks retrieved
- `RE_RANKING` — `0` off, `1` on
- `RERANK_K` — candidates retrieved before reranking down to `K`

The same variables are set in `load_test.sh` for the serving run.