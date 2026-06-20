import argparse
import json
import os
import time

import yaml
from datasets import Dataset

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
    model_name = config['ollama_model']

    if not args.skip_gen:
        print(f"--- Generating Synthetic Data (n={n}) ---")

        generation_log = {
            "sst2": {
                # ...
                "total_generation_time_seconds": 0.0,
                "attempts": 0,
                "total_examples_generated": 0,
            },
            "snli": {
                # ...
                "total_generation_time_seconds": 0.0,
                "attempts": 0,
                "total_examples_generated": 0,
            },
            "total_generation_time_seconds": 0.0,
            "attempts": 0,
            "total_examples_generated": 0,
        }

        start_time = time.perf_counter()

        try:
            # SST2
            sst2_neg_data, neg_generation_log = generate_synthetic_data.generate_synthetic_sst2(model_name, config['generation_seed'], "negative", 0, n)
            s_neg = Dataset.from_list(sst2_neg_data)
            sst2_pos_data, pos_generation_log = generate_synthetic_data.generate_synthetic_sst2(model_name, config['generation_seed'], "positive", 1, n)
            s_pos = Dataset.from_list(sst2_pos_data)

            # Write generated datasets to disk
            dataset_manager.write_dataset_to_disk(f"{data_dir}/sst2/synthetic_negative.jsonl", s_neg)
            dataset_manager.write_dataset_to_disk(f"{data_dir}/sst2/synthetic_positive.jsonl", s_pos)

            # Class Logs
            generation_log["sst2"]["negative"] = neg_generation_log
            generation_log["sst2"]["positive"] = pos_generation_log

            # Task Logs
            generation_log["sst2"]["total_generation_time_seconds"] = time.perf_counter() - start_time
            generation_log["sst2"]["attempts"] = neg_generation_log["attempts"] + pos_generation_log["attempts"]
            generation_log["sst2"]["total_examples_generated"] = len(s_neg) + len(s_pos)

            snli_start_time = time.perf_counter()

            # SNLI
            snli_ent_data, ent_generation_log = generate_synthetic_data.generate_synthetic_snli(model_name, config['generation_seed'], "entails", 0, n)
            s_ent = Dataset.from_list(snli_ent_data)
            snli_con_data, con_generation_log = generate_synthetic_data.generate_synthetic_snli(model_name, config['generation_seed'], "contradicts", 1, n)
            s_con = Dataset.from_list(snli_con_data)
            snli_neu_data, neu_generation_log = generate_synthetic_data.generate_synthetic_snli(model_name, config['generation_seed'], "is neutral with respect to", 2, n)
            s_neu = Dataset.from_list(snli_neu_data)

            # Write generated datasets to disk
            dataset_manager.write_dataset_to_disk(f"{data_dir}/snli/synthetic_entailment.jsonl", s_ent)
            dataset_manager.write_dataset_to_disk(f"{data_dir}/snli/synthetic_contradiction.jsonl", s_con)
            dataset_manager.write_dataset_to_disk(f"{data_dir}/snli/synthetic_neutral.jsonl", s_neu)

            # Class Logs
            generation_log["snli"]["entails"] = ent_generation_log
            generation_log["snli"]["contradicts"] = con_generation_log
            generation_log["snli"]["neutral"] = neu_generation_log

            # Task Logs
            generation_log["snli"]["total_generation_time_seconds"] = time.perf_counter() - snli_start_time
            generation_log["snli"]["attempts"] = ent_generation_log["attempts"] + con_generation_log["attempts"] + neu_generation_log["attempts"]
            generation_log["snli"]["total_examples_generated"] = len(s_ent) + len(s_con) + len(s_neu)

            # Total Generation Logs
            generation_log["total_generation_time_seconds"] = time.perf_counter() - start_time
            generation_log["attempts"] = generation_log["sst2"]["attempts"] + generation_log["snli"]["attempts"]
            generation_log["total_examples_generated"] = generation_log["sst2"]["total_examples_generated"] + generation_log["snli"]["total_examples_generated"]

            # Define path to save results
            os.makedirs(data_dir, exist_ok=True)
            generation_log_file_path = os.path.join(data_dir, "synthetic_generation_log.json")

            # Save results to a JSON file
            with open(generation_log_file_path, "w") as f:
                json.dump(generation_log, f, indent=4)
        except Exception as e:
            print(f"Error during synthetic data generation: {e}")
            print(f"! Aborting training and evaluation pipeline due to generation failure.")
            return


    # --- Stage 2: Training & Evaluation ---
    print("\n--- Starting Training and Evaluation Pipeline ---")

    tokenizer = tokenize_data.get_tokenizer(config['model_name'])
    training_config = config['training_params']
    training_config['seed'] = config['seed']
    training_config['results_dir'] = config['paths']['results_dir']

    # Load human SST-2
    raw_sst2_human = dataset_manager.load_sst2_dataset()
    # Tokenize human SST-2
    tokenized_human_sst2 = tokenize_data.tokenize(raw_sst2_human, tokenizer, training_config['max_seq_length'])

    # Load human SNLI
    raw_snli_human = dataset_manager.load_snli_dataset()
    # Tokenize human SNLI
    tokenized_human_snli = tokenize_data.tokenize(raw_snli_human, tokenizer, training_config['max_seq_length'])

    task_mapping = {
        "sst2": {
            "tokenized_human": tokenized_human_sst2,
            "label_feature": tokenized_human_sst2["train"].features['label'],
            "columns_to_save": ["sentence", "label"]
        },
        "snli": {
            "tokenized_human": tokenized_human_snli,
            "label_feature": tokenized_human_snli["train"].features['label'],
            "columns_to_save": ["premise", "hypothesis", "label"]
        }
    }

    for experiment in config['experiments']:
        exp_name = experiment['name']
        task = experiment['task']
        data_type = experiment['data_type']
        num_labels = experiment['num_labels']

        if task not in task_mapping:
            print(f"Skipping experiment '{exp_name}' due to unknown task '{task}'.")
            continue

        task_config = task_mapping[task]
        tokenized_human = task_config['tokenized_human']
        label_feature = task_config['label_feature']

        eval_dataset = tokenized_human["validation"]
        train_dataset_source = None

        if data_type == "human":
            train_dataset_source = tokenized_human["train"]
        elif data_type in ("synthetic", "mixed"):
            dataset_dict = dataset_manager.load_dataset_from_disk(experiment['synthetic_data_paths'])
            tokenized_synth = tokenize_data.tokenize(
                dataset_dict, tokenizer, training_config['max_seq_length']
            )
            if data_type == "synthetic":
                train_dataset_source = tokenized_synth["train"].cast_column("label", label_feature)
            else:  # data_type == mixed
                train_dataset_source = [
                    tokenized_human["train"],
                    tokenized_synth["train"].cast_column("label", label_feature)
                ]
        # smaller subsets are also subsets of bigger subsets with same seed, so we can compare different sample sizes directly
        train_dataset = dataset_manager.stratified_sample(
            train_dataset_source, "label", config['sample_size_per_class'], seed=config['seed']
        )

        # Write train dataset to disk
        dataset_manager.write_dataset_to_disk(f"{data_dir}/{task}/{data_type}.jsonl", train_dataset.select_columns(task_config['columns_to_save']))

        if train_dataset is not None and eval_dataset is not None:
            print(f"\n--- Starting Experiment: {exp_name} ---")
            print(f"Training samples: {len(train_dataset)}")
            print(f"Evaluation samples: {len(eval_dataset)}")
            res = train.run_experiment(exp_name, train_dataset, eval_dataset, num_labels, config['model_name'], training_config)
            print(f"Experiment {exp_name} completed. Final evaluation results: {json.dumps(res, indent=4)}")

if __name__ == "__main__":
    main()
