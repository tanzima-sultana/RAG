
from fastapi import APIRouter

from . import state
from .schemas import QueryRequest, QueryResponse

from constants import DENSE, BM25, HYBRID
from src.retrieval import Retrieval
from src.eval_qa import EvalQA
from src.evaluation import Evaluation

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):

    ev = EvalQA(request.mock_run, request.mode, request.dataset_size, request.num_queries)
    eval_set = ev.build_eval_set(request.chunking_type, state.CHUNKS_MAP, min_chunk_size=100)

    ret = Retrieval(
        request.mock_run,
        request.mode,
        state.CHUNKS_MAP,
        eval_set,
        request.k,
        request.re_ranking,
        request.rerank_k,
        request.model_name,
        request.device,
    )

    if request.retrieval_type == DENSE:
        retrieved_output = ret.retrieval_dense(state.FAISS_INDEX, state.FAISS_IDS)
    elif request.retrieval_type == BM25:
        retrieved_output = ret.retrieval_bm25(state.BM25_INDEX, state.BM25_IDS)
    elif request.retrieval_type == HYBRID:
        retrieved_output = ret.retrieval_hybrid(state.FAISS_INDEX, state.FAISS_IDS, state.BM25_INDEX, state.BM25_IDS)
    else:
        retrieved_output = ret.retrieval_qdrant(state.QDRANT_NAME)

    use_llm_judge = False
    eval = Evaluation(request.mode, request.dataset_size, request.model_name, use_llm_judge)
    results = eval.evaluate(request.k, retrieved_output)

    return QueryResponse(results=results)