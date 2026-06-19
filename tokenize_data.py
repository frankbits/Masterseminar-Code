from datasets import DatasetDict
from transformers import AutoTokenizer, TokenizersBackend


def get_tokenizer(model_name="bert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_name)

def tokenize(dataset: DatasetDict, tokenizer: TokenizersBackend, max_seq_length: int) -> DatasetDict:
    """
    Tokenizes a dataset using the provided tokenizer and max sequence length.
    Supports single-sentence (SST2) and sentence-pair (SNLI) formats automatically.
    """
    # Spaltennamen des Datasets auslesen
    column_names = dataset.column_names[list(dataset.keys())[0]]  # Annahme: Alle Splits haben die gleichen Spalten

    def tokenize_fn(batch):
        # Fallunterscheidung anhand der Felder im Dataset
        if "sentence" in column_names:
            # SST2-Format (Einzelner Satz)
            return tokenizer(
                batch["sentence"],
                truncation=True,
                padding="max_length",
                max_length=max_seq_length
            )
        elif "premise" in column_names and "hypothesis" in column_names:
            # SNLI-Format (Satzpaar)
            return tokenizer(
                batch["premise"],
                batch["hypothesis"],
                truncation=True,
                padding="max_length",
                max_length=max_seq_length
            )
        else:
            raise ValueError(
                f"Dataset contains unknown columns: {column_names}. "
                "Expected either ('premise', 'hypothesis') or ('sentence')."
            )

    # Tokenisierung auf alle Splits anzuwenden
    tokenized_ds = dataset.map(
        tokenize_fn,
        batched=True,
    )
    return tokenized_ds