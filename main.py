import argparse

import yaml
from datasets import concatenate_datasets

import generate_synthetic_data
import load_datasets
import tokenize_data
import train
from file_util import save_data_to_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")
    parser.add_argument("--skip_gen", action="store_true", help="Skip the LLM generation step")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    data_dir = config['paths']['data_dir']
    n = config['sample_size_per_class']
    m = config['ollama_model']

    if not args.skip_gen:
        print(f"--- Generating Synthetic Data (n={n}) ---")
        # SST2
        s_neg = generate_synthetic_data.generate_synthetic_sst2(m, "negative", 0, n)
        s_pos = generate_synthetic_data.generate_synthetic_sst2(m, "positive", 1, n)
        save_data_to_file(f"{data_dir}/sst2/synthetic_negative.json", s_neg)
        save_data_to_file(f"{data_dir}/sst2/synthetic_positive.json", s_pos)

        # SNLI
        s_ent = generate_synthetic_data.generate_synthetic_snli(m, "entails", 0, n)
        s_con = generate_synthetic_data.generate_synthetic_snli(m, "contradicts", 1, n)
        s_neu = generate_synthetic_data.generate_synthetic_snli(m, "is neutral with respect to", 2, n)
        save_data_to_file(f"{data_dir}/snli/synthetic_entailment.json", s_ent)
        save_data_to_file(f"{data_dir}/snli/synthetic_contradiction.json", s_con)
        save_data_to_file(f"{data_dir}/snli/synthetic_neutral.json", s_neu)


    # --- Stage 2: Training & Evaluation ---
    print("\n--- Starting Training and Evaluation Pipeline ---")

    tokenizer = tokenize_data.get_tokenizer(config['model_name'])
    training_config = config['training_params']
    training_config['seed'] = config['seed']
    training_config['results_dir'] = config['paths']['results_dir']

    # Load human datasets once
    raw_sst2_human = load_datasets.load_sst2_dataset()
    tokenized_human_sst2 = raw_sst2_human.map(
        lambda x: tokenizer(x["sentence"], truncation=True, padding="max_length",
                            max_length=training_config['max_seq_length']),
        batched=True
    )
    # Get the full Features object for SST-2 from the human dataset
    sst2_label_feature = tokenized_human_sst2["train"].features['label']


    raw_snli_human = load_datasets.load_snli_dataset()
    tokenized_human_snli = raw_snli_human.map(
        lambda x: tokenizer(x["premise"], x["hypothesis"], truncation=True, padding="max_length",
                            max_length=training_config['max_seq_length']),
        batched=True
    )
    # Get the full Features object for SNLI from the human dataset
    snli_label_feature = tokenized_human_snli["train"].features['label']

    # Calculate total sizes for subsetting
    sst2_total_size = config['sample_size_per_class'] * 2  # 2 classes for SST-2
    snli_total_size = config['sample_size_per_class'] * 3  # 3 classes for SNLI

    for experiment in config['experiments']:
        exp_name = experiment['name']
        task = experiment['task']
        data_type = experiment['data_type']
        num_labels = experiment['num_labels']

        train_dataset = None
        eval_dataset = None

        if task == "sst2":
            eval_dataset = tokenized_human_sst2["validation"]
            if data_type == "human":
                train_dataset = tokenized_human_sst2["train"].shuffle(seed=config['seed']).select(range(sst2_total_size))
            elif data_type == "synthetic":
                tokenized_synth_sst2 = tokenize_data.prepare_sst2_dataset(
                    experiment['synthetic_data_paths'], tokenizer, training_config['max_seq_length']
                )
                train_dataset = tokenized_synth_sst2["train"].cast_column("label", sst2_label_feature)
            elif data_type == "mixed":
                tokenized_synth_sst2 = tokenize_data.prepare_sst2_dataset(
                    experiment['synthetic_data_paths'], tokenizer, training_config['max_seq_length']
                )
                mixed_sst2 = concatenate_datasets([
                    tokenized_human_sst2["train"].shuffle(seed=config['seed']).select(range(sst2_total_size)),
                    tokenized_synth_sst2["train"].cast_column("label", sst2_label_feature)
                ])
                train_dataset = mixed_sst2
        elif task == "snli":
            eval_dataset = tokenized_human_snli["validation"]
            if data_type == "human":
                train_dataset = tokenized_human_snli["train"].shuffle(seed=config['seed']).select(range(snli_total_size))
            elif data_type == "synthetic":
                tokenized_synth_snli = tokenize_data.prepare_snli_dataset(
                    experiment['synthetic_data_paths'], tokenizer, training_config['max_seq_length']
                )
                train_dataset = tokenized_synth_snli["train"].cast_column("label", snli_label_feature)
            elif data_type == "mixed":
                tokenized_synth_snli = tokenize_data.prepare_snli_dataset(
                    experiment['synthetic_data_paths'], tokenizer, training_config['max_seq_length']
                )
                mixed_snli = concatenate_datasets([
                    tokenized_human_snli["train"].shuffle(seed=config['seed']).select(range(snli_total_size)),
                    tokenized_synth_snli["train"].cast_column("label", snli_label_feature)
                ])
                train_dataset = mixed_snli
        if train_dataset is not None and eval_dataset is not None:
            train.run_experiment(exp_name, train_dataset, eval_dataset, num_labels, config['model_name'], training_config)

if __name__ == "__main__":
    main()
