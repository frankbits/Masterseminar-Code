import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, __version__ as transformers_version
from packaging import version
import json
import os


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0)
    }
def run_experiment(task_name, train_dataset, eval_dataset, num_labels, model_name, training_config):
    """
    Runs a training experiment for a given task.
    Re-initializes the model for each experiment to ensure a clean slate.
    """
    print(f"\n--- Starting {task_name} Experiment ---")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    # Create training arguments dictionary
    train_args_kwargs = {
        "output_dir": f"{training_config['results_dir']}/{task_name}",
        "num_train_epochs": training_config['num_epochs'],
        "per_device_train_batch_size": training_config['batch_size_train'],
        "per_device_eval_batch_size": training_config['batch_size_eval'],
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "seed": training_config['seed'],
        "learning_rate": float(training_config['learning_rate']),
        "report_to": "none"
    }

    # Version-safe warmup logic
    if version.parse(transformers_version) >= version.parse("5.0.0"):
        # In v5+, warmup_steps accepts a float ratio directly
        train_args_kwargs["warmup_steps"] = training_config['warmup_ratio']
    else:
        # Fallback for older versions: use the now deprecated warmup_ratio
        train_args_kwargs["warmup_ratio"] = training_config['warmup_ratio']

    current_training_args = TrainingArguments(**train_args_kwargs)

    trainer = Trainer(
        model=model,
        args=current_training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Get dataset sizes
    train_dataset_size = len(train_dataset)
    eval_dataset_size = len(eval_dataset)

    # Train the model
    trainer.train()

    # Extract evaluation results per epoch from the trainer's log history
    # Filter for entries that contain 'eval_loss' to get epoch-wise evaluation results
    epoch_eval_history = [
        log for log in trainer.state.log_history if "eval_loss" in log
    ]

    # Evaluate the model after training (this will be the best model if load_best_model_at_end is True)
    final_eval_results = trainer.evaluate()

    # Prepare data for quantitative analysis
    experiment_results = {
        "task_name": task_name,
        "model_name": model_name,
        "train_dataset_size": train_dataset_size,
        "eval_dataset_size": eval_dataset_size,
        "final_evaluation_results": final_eval_results,
        "epoch_evaluation_history": epoch_eval_history,
        "training_config": training_config
    }

    # Define path to save results
    os.makedirs(current_training_args.output_dir, exist_ok=True)
    results_file_path = os.path.join(current_training_args.output_dir, "experiment_results.json")

    # Save results to a JSON file
    with open(results_file_path, "w") as f:
        json.dump(experiment_results, f, indent=4)

    print(f"--- Finished {task_name} Experiment ---")
    print(f"Results saved to {results_file_path}")
    return final_eval_results