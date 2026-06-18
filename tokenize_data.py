from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer

import dataset_manager


def get_tokenizer(model_name="bert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_name)

def prepare_sst2_dataset(json_paths, tokenizer, max_seq_length):
    """
    Loads JSON files and returns a tokenized HuggingFace DatasetDict.
    """
    # TODO: laden in dataset_manager auslagern? und hier nicht pfade sondern datasets übergeben?
    dataset = Dataset.from_list([])  # Start with an empty dataset
    for path in json_paths:
        dataset = concatenate_datasets(
            [
                dataset,
                dataset_manager.load_dataset_from_disk(path)
            ]
        )

    def tokenize_fn(batch):
        return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=max_seq_length)

    tokenized_ds = dataset.map(tokenize_fn, batched=True)
    # Wrap in DatasetDict for architectural symmetry with human data
    return DatasetDict({"train": tokenized_ds})

def prepare_snli_dataset(json_paths, tokenizer, max_seq_length):
    """
    Loads SNLI JSON files and returns a tokenized HuggingFace DatasetDict.
    """
    # TODO: laden in dataset_manager auslagern? und hier nicht pfade sondern datasets übergeben?
    dataset = Dataset.from_list([])  # Start with an empty dataset
    for path in json_paths:
        dataset = concatenate_datasets(
            [
                dataset,
                dataset_manager.load_dataset_from_disk(path)
            ]
        )

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