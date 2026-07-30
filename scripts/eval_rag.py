import argparse
import time

from src.eval_qa import EvalQA
from src.retrieval import Retrieval
from src.evaluation import Evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="RAG Eval")

    parser.add_argument("--num_queries", type=int, default=5,
                         help="Num of eval questions")
    parser.add_argument("--k", type=int, default=3,
                         help="Top k to retrieve")
    parser.add_argument("--retrieval_type", type=str, required=True,
                         choices=["dense", "bm25", "hybrid"],
                         help="Type of retrieval")
    parser.add_argument("--re_ranking", type=int, default=0,
                         help="Reranking yes(1) or no(0)")
    parser.add_argument("--rerank_k", type=int, default=10,
                         help="Top k for reranking")

    return parser.parse_args()

if __name__ == "__main__":

    print("\n ------------- Eval RAG -------- \n")
    s1 = time.time()

    # ---------------- 1. args
    args = parse_args()
    print(args)

    num_queries = args.num_queries
    