import os
import pickle

from constants import FIXED_CHUNKING, SENTENCE_CHUNKING, SEMANTIC_CHUNKING, MAX_CHUNK_SIZE, FIXED_CHUNK_OVERLAP

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone

model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = model.tokenizer

# ----- Output schema
CHUNK_SCHEMA = StructType([
    StructField("doc_id", StringType(), False),
    StructField("chunk_id", StringType(), False),
    StructField("chunk_text", StringType(), False),
    StructField("chunk_size", IntegerType(), False),
    StructField("chunk_type", StringType(), False),
    StructField("timestamp", TimestampType(), False),
])

# 1. Fixed chunking

def chunk_partition_fixed(rows):
    #Runs once per partition. Tokenizer loaded once here, reused
    #for every doc in this partition — not reloaded per row.

    for row in rows:
        doc_id = row["doc_id"]
        text = row["text"]

        try:
            tokens = tokenizer.encode(text)
        except Exception as e:
            # malformed doc — log and skip, don't crash the partition
            print(f"[chunk_partition_fixed] doc_id={doc_id} failed encode: {e}")
            continue

        start = 0
        chunk_id = 0

        while start < len(tokens):
            end = start + MAX_CHUNK_SIZE
            # Get the tokens between start and end, chunk boundary
            chunk_tokens = tokens[start:end]
            # Get text from tokens. Decode
            chunk_text = tokenizer.decode(chunk_tokens)

            # Each row produces one tuple, matching the struct CHUNK_SCHEMA
            yield (
                doc_id,
                f"{doc_id}_{chunk_id}",
                chunk_text,
                len(chunk_tokens),
                FIXED_CHUNKING,
                datetime.now(timezone.utc)
            )

            start += MAX_CHUNK_SIZE - FIXED_CHUNK_OVERLAP
            chunk_id += 1

# ----- Get or compute chunks and save to disk
def compute_chunks(input_path, dataset_size, chunk_type):

    output_path = f"spark_chunks/{dataset_size}/{chunk_type}/"

    if os.path.exists(output_path):
        print("Loading chunks from disk : ", chunk_type)
        return output_path

    # Spark session
    spark = SparkSession.builder.appName("spark_chunking").getOrCreate()

    df = spark.read.parquet(input_path)

    fn = None
    if chunk_type == FIXED_CHUNKING:
        fn = chunk_partition_fixed
    
    chunked_rdd = df.rdd.mapPartitions(fn)
    chunked_df = spark.createDataFrame(chunked_rdd, schema=CHUNK_SCHEMA)

    chunked_df.write.mode("overwrite").parquet(output_path)

    spark.stop()

    return output_path


