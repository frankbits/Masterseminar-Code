from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from file_util import read_data_from_file


def get_tokenizer(model_name="bert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_name)

def prepare_sst2_dataset(json_paths, tokenizer, max_seq_length):
    """
    Loads JSON files and returns a tokenized HuggingFace DatasetDict.
    """
    all_data = []
    for path in json_paths:
        all_data.extend(read_data_from_file(path))

    dataset = Dataset.from_list(all_data)

    def tokenize_fn(batch):
        return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=max_seq_length)

    tokenized_ds = dataset.map(tokenize_fn, batched=True)
    # Wrap in DatasetDict for architectural symmetry with human data
    return DatasetDict({"train": tokenized_ds})

def prepare_snli_dataset(json_paths, tokenizer, max_seq_length):
    """
    Loads SNLI JSON files and returns a tokenized HuggingFace DatasetDict.
    """
    all_data = []
    for path in json_paths:
        all_data.extend(read_data_from_file(path))

    dataset = Dataset.from_list(all_data)

    def tokenize_fn(batch):
        return tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_length
        )

    tokenized_ds = dataset.map(tokenize_fn, batched=True)
    # Wrap in DatasetDict for architectural symmetry with human data
    return DatasetDict({"train": tokenized_ds})