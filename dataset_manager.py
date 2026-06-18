from pathlib import Path

from datasets import load_dataset, Dataset


def write_dataset_to_disk(filename: str, dataset: Dataset):
    """Writes a dataset to disk in JSONL format for qualitative analysis and later loading."""
    dataset.to_json(filename, orient="records", lines=True)


def load_dataset_from_disk(path: str) -> Dataset:
    """Loads a dataset from local filesystem."""
    return load_dataset("json", data_files=str(Path(path)), split="train") #TODO: with or without split?)

def load_sst2_dataset():
    """Loads the SST-2 dataset from Hugging Face."""
    return load_dataset("stanfordnlp/sst2")

def load_snli_dataset():
    """Loads the SNLI dataset from Hugging Face and filters out -1 labels."""
    snli = load_dataset("stanfordnlp/snli")
    return snli.filter(lambda x: x["label"] != -1)

# TODO: load synthetic dataset?
# TODO: load mixed dataset?