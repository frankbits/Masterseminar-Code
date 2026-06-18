import argparse

import yaml
from datasets import concatenate_datasets, Dataset

import generate_synthetic_data
import dataset_manager
import tokenize_data
import train


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
        s_neg = Dataset.from_list(generate_synthetic_data.generate_synthetic_sst2(m, "negative", 0, n))
        s_pos = Dataset.from_list(generate_synthetic_data.generate_synthetic_sst2(m, "positive", 1, n))
        dataset_manager.write_dataset_to_disk(f"{data_dir}/sst2/synthetic_negative.jsonl", s_neg)
        dataset_manager.write_dataset_to_disk(f"{data_dir}/sst2/synthetic_positive.jsonl", s_pos)

        # SNLI
        s_ent = Dataset.from_list(generate_synthetic_data.generate_synthetic_snli(m, "entails", 0, n))
        s_con = Dataset.from_list(generate_synthetic_data.generate_synthetic_snli(m, "contradicts", 1, n))
        s_neu = Dataset.from_list(generate_synthetic_data.generate_synthetic_snli(m, "is neutral with respect to", 2, n))
        dataset_manager.write_dataset_to_disk(f"{data_dir}/snli/synthetic_entailment.jsonl", s_ent)
        dataset_manager.write_dataset_to_disk(f"{data_dir}/snli/synthetic_contradiction.jsonl", s_con)
        dataset_manager.write_dataset_to_disk(f"{data_dir}/snli/synthetic_neutral.jsonl", s_neu)


    # --- Stage 2: Training & Evaluation ---
    print("\n--- Starting Training and Evaluation Pipeline ---")

    tokenizer = tokenize_data.get_tokenizer(config['model_name'])
    training_config = config['training_params']
    training_config['seed'] = config['seed']
    training_config['results_dir'] = config['paths']['results_dir']

    # Load human datasets once
    raw_sst2_human = dataset_manager.load_sst2_dataset()
    tokenized_human_sst2 = raw_sst2_human.map(
        lambda x: tokenizer(x["sentence"], truncation=True, padding="max_length",
                            max_length=training_config['max_seq_length']),
        batched=True
    )
    # Get the full Features object for SST-2 from the human dataset
    sst2_label_feature = tokenized_human_sst2["train"].features['label']


    raw_snli_human = dataset_manager.load_snli_dataset()
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
                train_dataset = tokenized_synth_sst2["train"].shuffle(seed=config['seed']).select(range(min(len(tokenized_synth_sst2["train"]), sst2_total_size))).cast_column("label", sst2_label_feature)
            elif data_type == "mixed":
                tokenized_synth_sst2 = tokenize_data.prepare_sst2_dataset(
                    experiment['synthetic_data_paths'], tokenizer, training_config['max_seq_length']
                )
                # Ensure 50/50 split: Take half of the total size from each source
                half_size = sst2_total_size // 2
                synth_available = len(tokenized_synth_sst2["train"])
                if synth_available < half_size:
                    # TODO: not sensible. should enforce same total number. ask user how to proceed in console? reask ai, cancel
                    print(f"Warning: Requested {half_size} synthetic samples, but only {synth_available} available. Reducing split to {synth_available} human / {synth_available} synthetic for fairness.")
                    target_half = synth_available
                else:
                    target_half = half_size

                human_part = tokenized_human_sst2["train"].shuffle(seed=config['seed']).select(range(target_half))
                synth_part = tokenized_synth_sst2["train"].shuffle(seed=config['seed']).select(range(target_half)).cast_column("label", sst2_label_feature)
                
                mixed_sst2 = concatenate_datasets([
                    human_part,
                    synth_part
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
                train_dataset = tokenized_synth_snli["train"].shuffle(seed=config['seed']).select(range(min(len(tokenized_synth_snli["train"]), snli_total_size))).cast_column("label", snli_label_feature)
            elif data_type == "mixed":
                tokenized_synth_snli = tokenize_data.prepare_snli_dataset(
                    experiment['synthetic_data_paths'], tokenizer, training_config['max_seq_length']
                )
                # Ensure 50/50 split
                half_size = snli_total_size // 2
                synth_available = len(tokenized_synth_snli["train"])
                if synth_available < half_size:
                    # TODO: not sensible. should enforce same total number. ask user how to proceed in console? reask ai, cancel
                    print(f"Warning: Requested {half_size} synthetic samples, but only {synth_available} available. Reducing split to {synth_available} human / {synth_available} synthetic for fairness.")
                    target_half = synth_available
                else:
                    target_half = half_size

                human_part = tokenized_human_snli["train"].shuffle(seed=config['seed']).select(range(target_half))
                synth_part = tokenized_synth_snli["train"].shuffle(seed=config['seed']).select(range(target_half)).cast_column("label", snli_label_feature)
                
                mixed_snli = concatenate_datasets([
                    human_part,
                    synth_part
                ])
                train_dataset = mixed_snli
        if train_dataset is not None and eval_dataset is not None:
            print(f"\n--- Starting Experiment: {exp_name} ---")
            print(f"Training samples: {len(train_dataset)}")
            print(f"Evaluation samples: {len(eval_dataset)}")
            res = train.run_experiment(exp_name, train_dataset, eval_dataset, num_labels, config['model_name'], training_config)
            print(f"Experiment {exp_name} completed. Final evaluation results: {res}")

if __name__ == "__main__":
    main()
