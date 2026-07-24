import os
import boto3
from datasets import load_dataset, load_from_disk

from config import LOCAL_DATASET, S3_DATASET, S3_BUCKET
from constants import SEED, PROCESSED_DATA_PATH

class Dataset:
    def __init__(self, size):
        self.s3_client = boto3.client("s3")
        self.size = size
    
    def transform(self, example):
        return {
            'doc_id': example['id'],
            'title' : example['title'],
            'text': example['text']
        }
    
    # ------------ AWS -------------
    def s3_key_exists(self, bucket, key):
        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except self.s3_client.exceptions.ClientError:
            return False

    def load_parquet_dataset_s3(self):
        s3_key = f"data/{self.size}.parquet"

        if self.s3_key_exists(S3_BUCKET, s3_key):
            print(f"s3://{S3_BUCKET}/{s3_key} already exists, loading from S3")
            return load_dataset("parquet", data_files=f"s3://{S3_BUCKET}/{s3_key}", split="train")

        print(f"s3://{S3_BUCKET}/{s3_key} not found, creating from raw Wikipedia data on S3")

        dataset_original = load_dataset("parquet", data_files={"train": S3_DATASET}, split="train")
        dataset_sample = dataset_original.filter(lambda x: len(x['text']) > 200).shuffle(seed=SEED).select(range(self.size))
        dataset = dataset_sample.map(self.transform, remove_columns=dataset_sample.column_names)

        print(f"Uploading to s3://{S3_BUCKET}/{s3_key}")
        dataset.to_parquet(f"s3://{S3_BUCKET}/{s3_key}")

        return dataset
