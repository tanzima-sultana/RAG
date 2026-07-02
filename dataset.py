import os
from datasets import load_dataset, load_from_disk
from config import SEED, DATASET, DATASET_SIZE, DATA_PATH

def transform(example):
    return {
        'doc_id': example['id'],
        'title' : example['title'],
        'text': example['text']
    }

def load_processed_dataset():
    data_path = f"{DATA_PATH}/{DATASET_SIZE}"
    if os.path.exists(data_path):
        #print("Loading data from disk")
        return load_from_disk(data_path)
    
    # ----- Original dataset
    #dataset_original = load_dataset(DATASET, "20231101.en", split="train")
    dataset_original = load_dataset("parquet", data_files={"train": DATASET}, split="train")
    print(dataset_original.features)
    #print(dataset_original.shape)
    #print(dataset_original[0])

    # ----- Dataset of size DATASET_SIZE
    dataset_sample = dataset_original.shuffle(seed=SEED).select(range(DATASET_SIZE))

    # ----- Save to disk
    dataset = dataset_sample.map(transform, remove_columns=dataset_sample.column_names)
    dataset.save_to_disk(data_path)
    return dataset


if __name__ == "__main__":
    dataset = load_processed_dataset()