from sentence_transformers import SentenceTransformer
import numpy as np
import os 
from pyspark.sql import SparkSession
import time 
from functools import partial
import pyarrow as pa
import pyarrow.parquet as pq


from config import S3_BUCKET
from constants import LOCAL, AWS

from src.dist import s3_utills
 
class Embedding:
    def __init__(self, num_partition, mode, model_name, dataset_size, device, chunking_type):
        self.num_partition = num_partition
        self.mode = mode
        self.model_name = model_name
        self.dataset_size = dataset_size
        self.device = device
        self.chunking_type = chunking_type

        self.path = f"embeddings/{mode}_{dataset_size}_{device}_{chunking_type}"

        if self.mode == AWS:
            self.path = f"s3://{S3_BUCKET}/" + self.path
    
    def is_exists(self):
        if self.mode == AWS:
            return s3_utills.s3_file_exists(self.path)
        else:
            return os.path.exists(self.path)
        
    def save(self, chunk_ids, doc_ids, titles, embeddings):
        table = pa.table({
                "chunk_id" : chunk_ids,
                "doc_id": doc_ids,
                "title": titles,
                "embedding": embeddings if isinstance(embeddings, list) else embeddings.tolist(),
            })
        
        if self.mode == LOCAL:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            pq.write_table(table, self.path)
        else:
            import s3fs
            fs = s3fs.S3FileSystem()
            with fs.open(self.path, "wb") as f:
                pq.write_table(table, f)
        
        print(f"Saved embeddings to {self.path}")

    def generate_embeddings(self, chunk_path, batch_size):
        
        if self.is_exists():
            print("loading embeddings from disk")
            return self.path

        try:
            builder = SparkSession.builder.appName("spark_embedding")
            if self.mode == LOCAL:
                builder = builder.master("local[*]")
            spark = builder.getOrCreate()

            df = spark.read.parquet(chunk_path)
            df = df.repartition(self.num_partition)
            rdd = df.rdd
            print(f"Partitions: {rdd.getNumPartitions()}")

            start = time.time()

            # Runs once per Spark partition. Model loaded once here, not once per row
            def embed_partition(rows, batch_size, device, model_name):
                model = SentenceTransformer(model_name, device=device)

                rows = list(rows)
                chunk_ids = [r["chunk_id"] for r in rows]
                doc_ids = [r["doc_id"] for r in rows]
                titles = [r["title"] for r in rows]
                texts = [r["chunk_text"] for r in rows]

                embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True)

                for chunk_id, doc_id, title, emb in zip(chunk_ids, doc_ids, titles, embeddings):
                    yield (chunk_id, doc_id, title, emb.tolist())

            embed_fn = partial(
                embed_partition,
                batch_size=batch_size,
                device=self.device,
                model_name=self.model_name,
            )

            results = rdd.mapPartitions(embed_fn).collect()

            elapsed = time.time() - start
            throughput = len(results) / elapsed
            print(f"Embedded {len(results)} chunks in {elapsed:.2f}s ({throughput:.1f} texts/sec)")

            chunk_ids = [r[0] for r in results]
            doc_ids = [r[1] for r in results]
            titles = [r[2] for r in results]
            embeddings = [r[3] for r in results]

            self.save(chunk_ids, doc_ids, titles, embeddings)
            spark.stop()
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            return None
        
        if not self.is_exists():
            print("Embedding write produced no output")
            return None
        
        return self.path