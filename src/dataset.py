import os
import boto3
from datasets import load_dataset, load_from_disk

from config import LOCAL_DATASET, S3_DATASET, S3_BUCKET
from constants import SEED, PROCESSED_DATA_PATH, LOCAL, AWS

from src.dist import s3_utills

class Dataset:
    def __init__(self, mode, dataset_size):
        self.s3_client = boto3.client("s3")
        self.mode = mode
        self.dataset_size = dataset_size

        self.path = f"{PROCESSED_DATA_PATH}/{self.dataset_size}.parquet"
        if self.mode == AWS:
            self.path = f"s3://{S3_BUCKET}/" + self.path

    
    def transform(self, example):
        return {
            'doc_id': example['id'],
            'title' : example['title'],
            'text': example['text']
        }
    
    def load_parquet_dataset_local(self):
        if os.path.exists(self.path):
            print("Loading parquet data from disk")
            return load_dataset("parquet", data_files=self.path, split="train")   

        dataset = None
        try:
            print("Create parquet dataset")
            dataset_original = load_dataset("parquet", data_files={"train": LOCAL_DATASET}, split="train")
            dataset_sample = dataset_original.filter(lambda x: len(x['text']) > 200).shuffle(seed=SEED).select(range(self.dataset_size))
            dataset = dataset_sample.map(self.transform, remove_columns=dataset_sample.column_names)
            dataset.to_parquet(self.path)
        except Exception as e:
            print(f"Dataset creation failed: {e}")
            return None

        if not os.path.exists(self.path):
            print("Dataset write produced no output")
            return None

        return dataset
    
    # ------------ AWS -------------
    
    def load_parquet_dataset_s3(self):
        if s3_utills.s3_file_exists(self.path):
            print(f"{self.path} already exists, loading from S3")
            return self.path

        try:
            print(f"{self.path} not found, creating from raw Wikipedia data on S3")
            dataset_original = load_dataset("parquet", data_files={"train": S3_DATASET}, split="train")
            dataset_sample = dataset_original.filter(lambda x: len(x['text']) > 200).shuffle(seed=SEED).select(range(self.dataset_size))
            dataset = dataset_sample.map(self.transform, remove_columns=dataset_sample.column_names)
            dataset.to_parquet(self.path)
        except Exception as e:
            print(f"Dataset creation failed: {e}")
            return None

        if not s3_utills.s3_file_exists(self.path):
            print("Dataset write produced no output")
            return None

        return self.path

