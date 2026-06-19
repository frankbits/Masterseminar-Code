from pathlib import Path

from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict


def write_dataset_to_disk(filename: str, dataset: Dataset):
    """Writes a dataset to disk in JSONL format for qualitative analysis and later loading."""
    dataset.to_json(filename, orient="records", lines=True)


def load_dataset_from_disk(paths: str | list[str]) -> DatasetDict :
    """Loads a dataset from local filesystem."""
    if isinstance(paths, str):
        # `load_dataset()` automatically puts flat lists into "train"-split
        return load_dataset("json", data_files=str(Path(paths)))

    # Alle Dateien einzeln als DatasetDict laden
    loaded_dicts = [
        # `load_dataset()` automatically puts flat lists into "train"-split
        load_dataset("json", data_files=str(Path(path)))
        for path in paths
    ]

    # Alle vorkommenden Split-Namen (z.B. 'train', 'test') dynamisch sammeln
    all_splits = set()
    for d in loaded_dicts:
        all_splits.update(d.keys())

    # Für jeden Split die Datasets aus allen Dateien verknüpfen
    combined_splits = {}
    for split in all_splits:
        # Nur Datasets einbeziehen, die diesen Split auch tatsächlich besitzen
        datasets_to_combine = [d[split] for d in loaded_dicts if split in d]
        combined_splits[split] = concatenate_datasets(datasets_to_combine)

    return DatasetDict(combined_splits)

def load_sst2_dataset() -> DatasetDict:
    """Loads the SST-2 dataset from Hugging Face."""
    return load_dataset("stanfordnlp/sst2")

def load_snli_dataset() -> DatasetDict:
    """Loads the SNLI dataset from Hugging Face and filters out -1 labels."""
    snli = load_dataset("stanfordnlp/snli")
    return snli.filter(lambda x: x["label"] != -1)

def stratified_sample(datasets: Dataset | list[Dataset], label_column: str, n_per_class: int, seed: int) -> Dataset:
    if isinstance(datasets, Dataset):
        datasets = [datasets]
    n_datasets = len(datasets)
    if n_per_class % n_datasets != 0:
        raise ValueError(f"`sample_size_per_class` ({n_per_class}) must be divisible by the number of datasets ({n_datasets}) for mixed sampling.")
    n_per_class_per_dataset = n_per_class // n_datasets

    all_classes = set()
    for ds in datasets:
        all_classes.update(ds[label_column])
    classes = sorted(all_classes)

    parts = []
    for c in classes:
        for ds_idx, dataset in enumerate(datasets):
            class_indices = [i for i, label in enumerate(dataset[label_column]) if label == c]

            # Prüfen, ob genug Daten vorhanden sind
            if len(class_indices) < n_per_class_per_dataset:
                raise ValueError(
                    f"Datensatz an Index {ds_idx} hat für Klasse {c} nur {len(class_indices)} Beispiele. "
                    f"Erforderlich für die faire Aufteilung sind aber {n_per_class_per_dataset}."
                )

            class_subset = dataset.select(class_indices).shuffle(seed=seed)
            parts.append(class_subset.select(range(n_per_class_per_dataset)))

    combined = concatenate_datasets(parts)
    return combined.shuffle(seed=seed)  # shuffle again so classes aren't grouped in training order