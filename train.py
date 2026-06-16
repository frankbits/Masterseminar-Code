import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, __version__ as transformers_version
from packaging import version


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro")
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

    trainer.train()
    print(f"--- Finished {task_name} Experiment ---")
    return trainer.evaluate()