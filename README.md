# Masterseminar-Code

This repository contains a Python pipeline for loading, generating, tokenizing, preparing training data and running the training runs for different experiments.

## Quick start

Install the Python dependencies:

```shell
python -m pip install -r requirements.txt
```

Run the pipeline

```shell
python main.py
```

## Pipeline-Options

### main.py

```
usage: main.py [-h] [--config CONFIG] [--skip_gen]

options:
  -h, --help       show this help message and exit
  --config CONFIG  Path to the configuration file
  --skip_gen       Skip the LLM generation step
```

### config.yaml

Example-Configuration:

```yaml
project_name: "master_seminar_comparison"
seed: 42
model_name: "bert-base-uncased"

# The number of samples per class for synthetic generation
# and for the human-annotated baseline subsets.
sample_size_per_class: 500

ollama_model: "llama3.1"

training_params:
  num_epochs: 3
  batch_size_train: 16
  batch_size_eval: 32
  learning_rate: 2e-5
  warmup_ratio: 0.1
  max_seq_length: 128

paths:
  data_dir: "./data"
  results_dir: "./results"

experiments:
  - name: "SST2_Human"
    task: "sst2"
    data_type: "human"
    num_labels: 2
    # human_data_source: "stanfordnlp/sst2" # Not strictly needed here as it's loaded once globally

  - name: "SST2_Synthetic"
    task: "sst2"
    data_type: "synthetic"
    num_labels: 2
    synthetic_data_paths:
      - "./data/sst2/synthetic_negative.json"
      - "./data/sst2/synthetic_positive.json"

  - name: "SST2_Mixed"
    task: "sst2"
    data_type: "mixed"
    num_labels: 2
    # human_data_source: "stanfordnlp/sst2"
    synthetic_data_paths:
      - "./data/sst2/synthetic_negative.json"
      - "./data/sst2/synthetic_positive.json"

  - name: "SNLI_Human"
    task: "snli"
    data_type: "human"
    num_labels: 3
    # human_data_source: "stanfordnlp/snli"

  - name: "SNLI_Synthetic"
    task: "snli"
    data_type: "synthetic"
    num_labels: 3
    synthetic_data_paths:
      - "./data/snli/synthetic_entailment.json"
      - "./data/snli/synthetic_contradiction.json"
      - "./data/snli/synthetic_neutral.json"

  - name: "SNLI_Mixed"
    task: "snli"
    data_type: "mixed"
    num_labels: 3
    # human_data_source: "stanfordnlp/snli"
    synthetic_data_paths:
      - "./data/snli/synthetic_entailment.json"
      - "./data/snli/synthetic_contradiction.json"
      - "./data/snli/synthetic_neutral.json"
```