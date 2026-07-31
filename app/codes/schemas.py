
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

class QueryResponse(BaseModel):
    dataset_size: int
    k: int
    num_questions: int
    recall: float
    precision: float
    mrr: float
    sas: float
    avg_faithfulness: float
    avg_relevancy: float
    avg_ans_lmm_correctness: float
    total_cost: float
    total_latency: float