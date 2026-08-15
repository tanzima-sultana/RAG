# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A RAG (retrieval-augmented generation) benchmarking pipeline. It builds chunking/embedding/index artifacts for a Wikipedia dataset at multiple sizes, then evaluates combinations of chunking strategy × index type × retrieval mode × reranking against a Claude-generated ground-truth eval set, scoring both retrieval metrics (recall, precision, MRR, SAS) and LLM-as-judge answer-quality metrics (faithfulness, relevancy, correctness). The same retrieval+generation path is also served via FastAPI and load-tested. See README.md for full architecture diagrams, benchmark results, and methodology — this file only covers what's needed to operate the code.

## Setup

- Python 3.9, PyTorch/CUDA, Docker (for Qdrant).
- `pip install -r requirements.txt`
- Copy `config.py.template` to `config.py` and fill in `DATASET` (glob path to the local parquet dataset) and `ANTHROPIC_MSG_API_KEY`. `config.py` is gitignored — never commit it or paste the key into code, logs, or commit messages.
- Qdrant runs in Docker (`docker run -d -p 6333:6333 -p 6334:6334 -v <dir>/qdrant:/qdrant/storage qdrant/qdrant`); see `qdrant_setup.md`.

## Commands

There is no linter or type checker configured in this repo. There is a pytest suite under `tests/` covering each phase of `scripts/build_rag.py` and `scripts/eval_rag.py` (dataset, chunking, embedding, indexing, vector DB, eval-set generation, retrieval dispatch, evaluation), plus two script-level orchestration tests. It mocks every heavy/network dependency (SentenceTransformer, CrossEncoder, Qdrant, Anthropic) via `tests/fakes.py`, so it runs in seconds with no GPU, Docker, or real API key — only a `config.py` (see Setup) needs to exist for imports to resolve. Run it with:

```bash
python3 -m pytest
```

`pytest.ini` points this at `tests/` and puts the repo root on `PYTHONPATH`. The `.claude/skills/git-commit` skill runs this suite before every commit and blocks the commit if it fails — beyond that, verification of the real pipeline is done by running it (optionally with `--mock_run 1`) and inspecting logs/output files.

**Local build + eval** (edit parameters at the top of the script first, then run):
```bash
./local_run.sh
```
This runs `scripts/build_rag.py` (chunk, embed, index, generate eval questions) followed by `scripts/eval_rag.py` (retrieve and score), writing `build_log.txt` and `eval_log.txt`.

**Run a single stage directly** instead of the full `local_run.sh`, e.g. to eval one config without rebuilding:
```bash
python3 scripts/eval_rag.py --mock_run 0 --mode local --device cuda \
  --model_name all-MiniLM-L6-v2 --dataset_size 20000 \
  --chunking_type fixed --index_type hnsw --retrieval_type hybrid \
  --num_queries 50 --k 5 --re_ranking 1 --rerank_k 20
```
Set `--mock_run 1` to exercise the pipeline with mocked Claude responses (no API cost) — use this when validating a code change before spending real API calls.

**Distributed (Spark) build**, e.g. for larger datasets or local Spark testing:
```bash
./dist_run.sh
```

**Serve + load test** (from `app/scripts`, in a separate terminal after Qdrant/build artifacts exist):
```bash
./run_server.sh     # starts FastAPI (uvicorn) on port 8001
./load_test.sh       # async httpx concurrency sweep against /query
./query_curl.sh       # one sample curl request
```

All shell scripts `source ~/pyenv/bin/activate` and set `PYTHONPATH` to the repo root — they assume that virtualenv path and must be run from the repo root (or with `cwd` set there for `run_server.sh`, which uses `$(pwd)`).

## Architecture

**Two-phase pipeline, connected by manifest files.** The build phase (`scripts/build_rag.py`) writes chunks, embeddings, FAISS/BM25/Qdrant indexes, and eval question sets to disk, then records their paths in `manifests/<dataset_size>_manifest.json`, keyed by chunking strategy. Every downstream consumer — `scripts/eval_rag.py` and `app/codes/state.py` (the FastAPI startup loader) — reads artifacts exclusively through this manifest rather than reconstructing paths itself. When adding a new artifact type or changing a naming scheme, update the manifest-writing code in the build phase and every manifest reader stays in sync automatically.

**`local` vs `aws` mode is threaded through almost everything.** Most modules and CLI scripts take a `mode` argument (`constants.LOCAL` / `constants.AWS`) that switches between local filesystem paths and `s3://` paths (bucket in `config.S3_BUCKET`). `src/local/` holds the local chunking/embedding implementation; `src/dist/` holds the Spark + S3 distributed equivalent (`src/dist/chunking_embedding.py`, `src/dist/s3_utills.py`), driven by `scripts/run_dist_rag.py` instead of `build_rag.py`. Shared logic (indexing, retrieval, evaluation, the Anthropic wrapper) lives directly under `src/` and is mode-agnostic.

**Config resolution order for a given run**: CLI args (parsed in each `scripts/*.py`, defaults visible in `parse_args()`) → `constants.py` (fixed enums like chunking/index/retrieval type strings, cost-per-token rates) → `config.py` (machine-specific secrets/paths: API key, dataset glob, S3 bucket — gitignored, not present until created from the template).

**Retrieval modes are combined via a common interface** in `src/retrieval.py`: dense (FAISS), sparse (BM25), hybrid (Reciprocal Rank Fusion of the two), and vectordb (Qdrant, an alternate backend to FAISS entirely — see `src/vector_db.py`). Optional cross-encoder reranking sits after retrieval and before generation, reordering the top-`rerank_k` candidates down to `k`.

**Chunk and result identity is `doc_id_chunk_id`.** Retrieval metrics (recall/precision/MRR) are scored by extracting `doc_id` from this composite ID and matching against the eval set's ground-truth document — not by exact chunk match — because chunk boundaries differ per chunking strategy and dataset size. Eval question sets are therefore generated separately per chunking strategy (`src/eval_qa.py`) and are not directly comparable across strategies (see README "Known Issues" for the implication).

**Answer-quality scoring is LLM-as-judge**, not the retrieval-metric code path: `src/evaluation.py` sends batched Claude Messages API calls (faithfulness, relevancy, correctness, each 0–1) plus a separate embedding-based semantic answer similarity (SAS) score. All generation and judging Claude calls go through the single wrapper in `src/anthropic_api.py`, which also computes per-call cost from `constants.INPUT_COST_PER_MTOK` / `OUTPUT_COST_PER_MTOK` — route any new Claude call through this wrapper rather than instantiating the SDK client directly, to keep cost tracking consistent.

**The FastAPI service (`app/codes/`) mirrors the eval-time retrieval path but loads state once.** `state.py` loads the manifest, indexes, chunks, eval set, embedding model, and cross-encoder into module-level globals at startup (given `mode`, `model_name`, `dataset_size`, `chunking_type`, `index_type` from env vars set in `run_server.sh`); `query.py` runs embed → retrieve → rerank → generate per request against that shared state; `fast_api.py` exposes it as `POST /query`; `schemas.py` defines the request/response Pydantic models. `concurrent_req.py` is the async load-test client driven by `load_test.sh`.

## Data and artifacts are not committed

`chunks/`, `embeddings/`, `index/`, `data/`, `evals/`, `eval_qa/`, build/eval logs, and the benchmark spreadsheets are all gitignored — they're large, regenerable outputs of a build run, not source. Manifests under `manifests/` (which just contain relative paths) are the only build-phase output that's expected to be readable without rerunning the build.
