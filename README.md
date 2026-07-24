GPU Acceleration — Local Pipeline Findings (1K docs)

Across the local RAG pipeline (chunking → embedding → indexing → retrieval), Embedding benefits most from GPU acceleration. Semantic chunking is a close second. All other stages show no meaningful GPU benefit.

Stage	GPU benefit	Reason
Chunking (fixed/sentence)	None (~1x)	Pure tokenizer work. No model forward pass.
Chunking (semantic)	~7.4x	Embeds every sentence for similarity comparison.
Embedding	~7–7.2x	Full transformer forward pass per chunk.
Indexing	N/A (CPU-only)	FAISS index build is not GPU-accelerated in this pipeline.

Chunking time, 1K docs, by chunking strategy

Chunking type	CPU	CUDA	Speedup
Fixed	6.0s	6.67s	~1x
Sentence	6.08s	6.1s	~1x
Semantic	156s	21s	~7.4x

Embedding time, 1K docs, by chunking strategy

Chunking type	CPU	CUDA	Speedup
Fixed	64.14s	8.95s	~7.2x
Sentence	61.8s	8.82s	~7.0x
Semantic	68.05s	9.54s	~7.1x

Takeaway

Fixed and sentence-aware chunking involve no model forward pass. GPU offers no benefit for these. Semantic chunking embeds every sentence to compute similarity. This gives it a ~7.4x GPU speedup, comparable to the embedding stage itself. Embedding time is consistently ~7x faster on GPU across all chunking strategies. GPU benefit scales with chunk or sentence count, not chunking method. Embedding is the highest-leverage target for GPU acceleration in this pipeline. If semantic chunking is used, chunking becomes a second priority. At larger scale (Week 5's 50K+ chunks), these stages are expected to dominate CPU wall-clock time. GPU allocation is therefore a priority infrastructure decision in the local-to-distributed pipeline.