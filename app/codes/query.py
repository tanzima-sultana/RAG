
from fastapi import APIRouter
import random

from . import state
from .schemas import QueryRequest, QueryResponse

from constants import DENSE, BM25, HYBRID
from src.retrieval import Retrieval
from src.evaluation import Evaluation

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    
    # From request
    mock_run = request.mock_run
    mode = request.mode
    device = request.device
    model_name = request.model_name
    dataset_size = request.dataset_size
    chunking_type = request.chunking_type
    index_type = request.index_type
    retrieval_type = request.retrieval_type
    num_queries = request.num_queries
    k = request.k
    re_ranking = request.re_ranking
    rerank_k = request.rerank_k

    # From state.py loaded at startup
    chunks_map = state.CHUNKS_MAP
    faiss_index = state.FAISS_INDEX
    faiss_ids = state.FAISS_IDS
    bm25_index = state.BM25_INDEX
    bm25_ids = state.BM25_IDS
    qdrant_name = state.QDRANT_NAME
    eval_set = state.EVAL_SET
    model = state.MODEL
    cross_encoder = state.CROSS_ENCODER

    # ----------- 1. Evaluation Qus-Ans Set
    eval_set = random.sample(state.EVAL_SET, num_queries)

    # ----------- 2. Retrival
    ret1 = Retrieval(mock_run, mode, chunking_type, chunks_map, eval_set, k, re_ranking, rerank_k, model, cross_encoder)
    retrieved_output = None

    if retrieval_type == DENSE:
        retrieved_output = ret1.retrieval_dense(faiss_index, faiss_ids)
    elif retrieval_type == BM25:
        retrieved_output = ret1.retrieval_bm25(bm25_index, bm25_ids)
    elif retrieval_type == HYBRID:
        retrieved_output = ret1.retrieval_hybrid(faiss_index, faiss_ids, bm25_index, bm25_ids)
    else:
        # Qdrant
        collection_name = qdrant_name
        retrieved_output = ret1.retrieval_qdrant(collection_name)

    return QueryResponse(results=retrieved_output)