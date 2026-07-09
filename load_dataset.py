import os
from datasets import load_dataset, load_from_disk
from constants import SEED, DATASET, DATASET_SIZE, DATA_PATH

def transform(example):
    return {
        'doc_id': example['id'],
        'title' : example['title'],
        'text': example['text']
    }

def load_parquet_dataset():
    data_path = f"{DATA_PATH}/{DATASET_SIZE}.parquet"
    if os.path.exists(data_path):
        print("Loading parquet data from disk")
        return load_dataset("parquet", data_files=data_path, split="train")   

    print("Craete parquet dataset")
    dataset_original = load_dataset("parquet", data_files={"train": DATASET}, split="train")
    dataset_sample = dataset_original.filter(lambda x: len(x['text']) > 5000).shuffle(seed=SEED).select(range(DATASET_SIZE))
    dataset = dataset_sample.map(transform, remove_columns=dataset_sample.column_names)
    dataset.to_parquet(data_path)   

    return dataset

def get_sample_parquet(dataset_size):
    full_path = f"{DATA_PATH}/{DATASET_SIZE}.parquet"
    sample_path = f"{DATA_PATH}/{dataset_size}.parquet"

    if os.path.exists(sample_path):
        return sample_path

    full_dataset = load_dataset("parquet", data_files=full_path, split="train")
    subset = full_dataset.select(range(dataset_size))
    subset.to_parquet(sample_path)

    return sample_path