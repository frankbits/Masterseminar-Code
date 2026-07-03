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

# Generation-Seed for reproducibility.
# This ensures that the same random samples are generated across different runs of the experiments.
generation_seed: 42

# Data-Seed for reproducibility.
# Seed for shuffling and splitting the human-annotated dataset into training and evaluation sets.
# This ensures that the same data splits are used across different runs of the experiments.
data_seed: 42

# Training-Seeds for reproducibility.
# This ensures that the same random initialization and data shuffling occur across different runs of the experiments using the same seed.
# This should be set to three different values (e.g.: [42, 123, 7]) to run experiments with different random seeds and average the results for more robust conclusions.
seeds: [42, 123, 7]

# The base model used for generating synthetic data.
# This should be a model that is capable of few-shot prompting and can generate high-quality text.
ollama_model: "llama3.1"
# The downstream model to be trained and evaluated on the human-annotated and synthetic datasets.
# This should be a small model that is intended to be fine-tuned for the downstream tasks, such as sentiment analysis or natural language inference.
model_name: "bert-base-uncased"

# The number of samples per class for synthetic or human-annotated data.
# This is used to control the size of the training dataset for each experiment.
# has to be divisible by the number of datasets used for the mixed setting (human and synthetic -> 2 datasets, so sample_size_per_class has to be divisible by 2).
sample_size_per_class: 1000 #TODO: Allow multiple values for this parameter to automatically run experiments with different dataset sizes.
# The number of samples to generate in each batch when generating synthetic data.
# The `sample_size_per_class` gets divided into batches of this size for generation.
generation_batch_size: 50
# The maximum number of attempts to automatically generate a valid sample for each class during synthetic data generation.
generation_max_attempts: 3

# Training parameters for the fine-tuning of the downstream model on the human-annotated and synthetic datasets.
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

  - name: "SST2_Synthetic"
    task: "sst2"
    data_type: "synthetic"
    num_labels: 2
    synthetic_data_paths:
      - "./data/sst2/synthetic_negative.jsonl"
      - "./data/sst2/synthetic_positive.jsonl"

  - name: "SST2_Mixed"
    task: "sst2"
    data_type: "mixed"
    num_labels: 2
    synthetic_data_paths:
      - "./data/sst2/synthetic_negative.jsonl"
      - "./data/sst2/synthetic_positive.jsonl"

  - name: "SNLI_Human"
    task: "snli"
    data_type: "human"
    num_labels: 3

  - name: "SNLI_Synthetic"
    task: "snli"
    data_type: "synthetic"
    num_labels: 3
    synthetic_data_paths:
      - "./data/snli/synthetic_entailment.jsonl"
      - "./data/snli/synthetic_contradiction.jsonl"
      - "./data/snli/synthetic_neutral.jsonl"

  - name: "SNLI_Mixed"
    task: "snli"
    data_type: "mixed"
    num_labels: 3
    synthetic_data_paths:
      - "./data/snli/synthetic_entailment.jsonl"
      - "./data/snli/synthetic_contradiction.jsonl"
      - "./data/snli/synthetic_neutral.jsonl"
```

## Running, aggregating and evaluating experiments

1. Run the pipeline with the desired configuration:

Set the `sample_size_per_class` in the `config.yaml` file to the desired biggest value (e.g., 1000) and run:

```shell
python main.py
```

2. Aggregate the results from all defined experiments and seeds:

```shell
python scripts/aggregate_results.py
```

3. Copy the used synthetic data and the aggregated results to a separate folder:

Copy the `data` and `results` folders to a separate folder using the following folder structure:

```
root/
├── results/
│   ├── {sample_size_per_class}_seeds-{seed1-seed2-...}/
│   │   ├── data/
│   │   └── results/
│   ├── {sample_size_per_class2}_seeds-{seed1-seed2-...}/
│   │   ├── data/
│   │   └── results/
│   └── ...
```

4. Repeat for all smaller `sample_size_per_class`-values:

Set the `sample_size_per_class` in the `config.yaml` file to the desired smaller value (e.g., 500) and run, while skipping the generation step (reusing the already generated synthetic data):

```shell
python main.py --skip_gen
```

Aggregate the results:

```shell
python scripts/aggregate_results.py
```

Repeat this process for all desired dataset sizes.

5. In the external folder, generate the latex tables and plots:

You have a folder structure like this:
```
root/
├── results/
│   ├── 1000_seeds-42-123-7/
│   │   ├── data/
│   │   └── results/
│   ├── 500_seeds-42-123-7/
│   │   ├── data/
│   │   └── results/
│   └── ...
```

Copy the script to generate the latex tables and plots into the external folder and run it:

```shell
python generate_latex_tables_and_plots.py
```

The script will generate the latex tables and plots in the `latex_figures` folder, based on the aggregated results from all experiments and seeds in the `results` folder.

6. Use the data, results and generated latex tables and plots for evaluation and discussion of the experiments.