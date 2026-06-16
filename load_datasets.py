from datasets import load_dataset

def load_sst2_dataset():
    """Loads the SST-2 dataset from Hugging Face."""
    return load_dataset("stanfordnlp/sst2")

def load_snli_dataset():
    """Loads the SNLI dataset from Hugging Face and filters out -1 labels."""
    snli = load_dataset("stanfordnlp/snli")
    return snli.filter(lambda x: x["label"] != -1)