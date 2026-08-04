
from pydantic import BaseModel
from typing import Literal

class QueryRequest(BaseModel):
    mock_run: int = 1
    mode: Literal["local", "aws"]
    device: Literal["cpu", "cuda"]
    model_name: str
    dataset_size: int
    chunking_type: Literal["fixed", "sentence", "semantic"]
    index_type: Literal["flatip", "ivf", "hnsw", "vectordb"]
    retrieval_type: Literal["dense", "bm25", "hybrid"]
    num_queries: int = 5
    k: int = 3
    re_ranking: int = 0
    rerank_k: int = 10

class QueryResult(BaseModel):
    retrieval_type: str
    chunk_type: str
    chunk_id: str
    retrieved_chunk_ids: list
    retrieved_chunk_texts: list
    qus: str
    context: str
    generated_ans: str
    ground_truth_ans: str
    k: int
    cost: float
    latency: float

class QueryResponse(BaseModel):
    results: list[QueryResult]