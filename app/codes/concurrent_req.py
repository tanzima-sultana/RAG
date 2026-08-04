
import asyncio
import time
import argparse
import httpx
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_url", default="http://127.0.0.1:8001")
    p.add_argument("--mock_run", type=int, default=1)
    p.add_argument("--mode", default="local")
    p.add_argument("--device", default="cuda")
    p.add_argument("--model_name", default="all-MiniLM-L6-v2")
    p.add_argument("--dataset_size", type=int, default=50000)
    p.add_argument("--chunking_type", default="fixed")
    p.add_argument("--index_type", default="flatip")
    p.add_argument("--retrieval_type", default="hybrid")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--re_ranking", type=int, default=1)
    p.add_argument("--rerank_k", type=int, default=20)
    p.add_argument("--concurrency_levels", type=int, nargs="+", default=[5, 10])
    p.add_argument("--requests_per_level", type=int, default=5)  # multiplied by concurrency
    return p.parse_args()

async def one_request(client, base_url, config):
    t0 = time.perf_counter()
    try:
        r = await client.post(f"{base_url}/query", json=config)
        ok = r.status_code == 200
        if not ok:
            print(f"FAILED status={r.status_code} body={r.text}")
    except Exception as e:
        ok = False
        print(f"EXCEPTION: {e}")
    return (time.perf_counter() - t0) * 1000, ok

async def run_load_test(base_url, config, concurrency, total_requests):
    async with httpx.AsyncClient(timeout=60) as client:
        results = []
        wall_start = time.perf_counter()
        for i in range(0, total_requests, concurrency):
            batch = [one_request(client, base_url, config) for _ in range(min(concurrency, total_requests - i))]
            results.extend(await asyncio.gather(*batch))
        wall_time_s = time.perf_counter() - wall_start

    latencies = [r[0] for r in results]
    errors = [r for r in results if not r[1]]

    return {
        "concurrency": concurrency,
        "p50": np.percentile(latencies, 50),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
        "throughput_rps": total_requests / wall_time_s,
        "error_rate": len(errors) / total_requests,
    }

if __name__ == "__main__":
    args = parse_args()
    config = {
        "mock_run": args.mock_run,
        "mode": args.mode,
        "device": args.device,
        "model_name": args.model_name,
        "dataset_size": args.dataset_size,
        "chunking_type": args.chunking_type,
        "index_type": args.index_type,
        "retrieval_type": args.retrieval_type,
        "num_queries": 1,
        "k": args.k,
        "re_ranking": args.re_ranking,
        "rerank_k": args.rerank_k,
    }
    for c in args.concurrency_levels:
        print(asyncio.run(run_load_test(args.base_url, config, c, c * args.requests_per_level)))