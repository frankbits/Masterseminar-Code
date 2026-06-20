"""
aggregate_results.py

Durchsucht den `results/`-Ordner nach allen `experiment_results.json`-Dateien
(eine pro Bedingung, z.B. SST2_Human, SNLI_Mixed, ...), fasst die Kennzahlen
zusammen und erzeugt:

  1. results/all_results.csv          -> finale Metriken pro Bedingung (Tabelle)
  2. results/all_results.md           -> dieselbe Tabelle als Markdown (z.B. für ACL-Paper)
  3. results/epoch_history_long.csv   -> Metriken pro Epoche, "long format" (für eigene Plots)
  4. results/plots/f1_by_condition.png        -> Balkendiagramm: Macro F1 je Task/Setting
  5. results/plots/accuracy_by_condition.png  -> Balkendiagramm: Accuracy je Task/Setting
  6. results/plots/<task>_learning_curves.png -> Lernkurven (eval_f1 über Epochen) je Task

Nutzung:
    python aggregate_results.py [--results-dir results] [--out-dir results]

Voraussetzungen:
    pip install pandas matplotlib --break-system-packages   # falls noch nicht installiert
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_synthetic_log(log_path):
    with open(log_path, 'r') as f:
        return json.load(f)


def flatten_synthetic_log(log_data):
    """
    Flacht den synthetic_generation_log.json Inhalt in zwei DataFrames auf.

    Erwartete Struktur (neu, passend zu generate_synthetic_data()):
      {
        "sst2": {
          "negative": {"config": {...}, "batches": [...], "total_retries": int,
                       "aborted": bool, "generation_time_seconds": float, "items_generated": int},
          "positive": {...},
          "generation_time_seconds": float, "total_retries": int,
          "items_generated": int, "aborted": bool
        },
        "snli": {... gleiche Struktur mit entails/contradicts/neutral ...},
        "generation_time_seconds": float, "total_retries": int,
        "items_generated": int, "aborted": bool
      }
    """
    totals_rows = []
    task_totals_rows = []
    tasks_rows = []

    totals_rows.append({
        "task": "TOTAL",
        "generation_time_seconds": log_data.get("generation_time_seconds"),
        "total_retries": log_data.get("total_retries"),
        "items_generated": log_data.get("items_generated"),
        "aborted": log_data.get("aborted"),
    })

    for key, value in log_data.items():
        if not isinstance(value, dict):
            continue

        task = key
        task_data = value
        task_totals_rows.append({
            "task": task,
            "generation_time_seconds": task_data.get("generation_time_seconds"),
            "total_retries": task_data.get("total_retries"),
            "items_generated": task_data.get("items_generated"),
            "aborted": task_data.get("aborted"),
        })
        for subclass, subclass_data in task_data.items():
            # per-class logs are identifiable by having a "config" key
            if isinstance(subclass_data, dict) and "config" in subclass_data:
                cfg = subclass_data.get("config", {})
                batches = subclass_data.get("batches", [])
                tasks_rows.append({
                    "task": task,
                    "subclass": subclass,
                    "generation_time_seconds": subclass_data.get("generation_time_seconds"),
                    "total_retries": subclass_data.get("total_retries"),
                    "items_generated": subclass_data.get("items_generated"),
                    "items_requested": cfg.get("n"),
                    "batch_size": cfg.get("batch_size"),
                    "num_batches": len(batches),
                    "aborted": subclass_data.get("aborted"),
                })

    task_totals_rows.append({
        "task": "",
        "generation_time_seconds": "",
        "total_retries": "",
        "items_generated": "",
        "aborted": "",
    })
    task_totals_rows.extend(totals_rows)
    task_totals_df = pd.DataFrame(task_totals_rows)
    tasks_df = pd.DataFrame(tasks_rows)
    return task_totals_df, tasks_df


def find_result_files(results_dir: Path):
    """Findet alle experiment_results.json Dateien unterhalb von results_dir."""
    files = sorted(results_dir.glob("*/experiment_results.json"))
    if not files:
        # Fallback: rekursiv suchen, falls die Struktur tiefer verschachtelt ist
        files = sorted(results_dir.rglob("experiment_results.json"))
    return files


def parse_condition_name(folder_name: str):
    """
    Erwartet Ordnernamen wie 'SST2_Human', 'SNLI_Mixed', 'SST2_Synthetic'.
    Gibt (task, setting) zurück, z.B. ('SST2', 'Human').
    """
    parts = folder_name.split("_")
    if len(parts) >= 2:
        task = parts[0]
        setting = "_".join(parts[1:])
    else:
        task, setting = folder_name, "unknown"
    return task, setting


def load_all_results(results_dir: Path):
    """Lädt alle experiment_results.json und liefert zwei DataFrames:
    - summary_df: eine Zeile pro Bedingung (finale Metriken)
    - history_df: eine Zeile pro (Bedingung, Epoche) (Lernkurven)
    """
    files = find_result_files(results_dir)
    if not files:
        raise FileNotFoundError(
            f"Keine experiment_results.json Dateien unter '{results_dir}' gefunden."
        )

    summary_rows = []
    history_rows = []

    for file_path in files:
        condition_folder = file_path.parent.name
        task, setting = parse_condition_name(condition_folder)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        final = data.get("final_evaluation_results", {})
        cfg = data.get("training_config", {})

        summary_rows.append(
            {
                "condition": condition_folder,
                "task": task,
                "setting": setting,
                "seed": cfg.get("seed"),
                "train_dataset_size": data.get("train_dataset_size"),
                "eval_dataset_size": data.get("eval_dataset_size"),
                "eval_loss": final.get("eval_loss"),
                "eval_accuracy": final.get("eval_accuracy"),
                "eval_f1": final.get("eval_f1"),
                "eval_precision": final.get("eval_precision"),
                "eval_recall": final.get("eval_recall"),
                "epoch": final.get("epoch"),
            }
        )

        for entry in data.get("epoch_evaluation_history", []):
            history_rows.append(
                {
                    "condition": condition_folder,
                    "task": task,
                    "setting": setting,
                    "seed": cfg.get("seed"),
                    "epoch": entry.get("epoch"),
                    "eval_loss": entry.get("eval_loss"),
                    "eval_accuracy": entry.get("eval_accuracy"),
                    "eval_f1": entry.get("eval_f1"),
                    "eval_precision": entry.get("eval_precision"),
                    "eval_recall": entry.get("eval_recall"),
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(["task", "setting", "seed"])
    history_df = pd.DataFrame(history_rows).sort_values(["task", "setting", "seed", "epoch"])
    return summary_df, history_df


def aggregate_across_seeds(summary_df: pd.DataFrame):
    """Mittelwert + Standardabweichung über die Seeds je (task, setting)."""
    metrics = [c for c in ["eval_accuracy", "eval_f1", "eval_precision", "eval_recall", "eval_loss"]
               if c in summary_df.columns]
    agg = (
        summary_df.groupby(["task", "setting"])[metrics]
        .agg(["mean", "std"])
    )
    agg.columns = ["_".join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index()
    return agg


def save_tables(summary_df: pd.DataFrame, agg_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path_csv = out_dir / "all_results.csv"
    summary_df.to_csv(summary_path_csv, index=False)

    agg_path_csv = out_dir / "all_results_aggregated.csv"
    agg_df.to_csv(agg_path_csv, index=False)

    # Markdown-Tabelle (gut zum Copy-Paste fürs Paper)
    md_path = out_dir / "all_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("## Finale Metriken pro Run (alle Seeds)\n\n")
        f.write(summary_df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## Aggregiert über Seeds (Mittelwert ± Std)\n\n")
        f.write(agg_df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")

    print(f"Tabelle gespeichert: {summary_path_csv}")
    print(f"Aggregierte Tabelle gespeichert: {agg_path_csv}")
    print(f"Markdown-Tabelle gespeichert: {md_path}")


def plot_bar_by_condition(agg_df: pd.DataFrame, metric: str, out_path: Path, title: str):
    """Balkendiagramm: Mittelwert + Fehlerbalken (Std) je Task, gruppiert nach Setting."""
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in agg_df.columns:
        print(f"Überspringe Plot '{title}': Spalte '{mean_col}' fehlt.")
        return

    tasks = sorted(agg_df["task"].unique())
    settings = sorted(agg_df["setting"].unique())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.8 / max(len(settings), 1)
    x_base = range(len(tasks))

    for i, setting in enumerate(settings):
        means, stds = [], []
        for task in tasks:
            row = agg_df[(agg_df["task"] == task) & (agg_df["setting"] == setting)]
            if row.empty:
                means.append(0)
                stds.append(0)
            else:
                means.append(row[mean_col].values[0])
                stds.append(row[std_col].values[0] if std_col in row else 0)
        positions = [x + i * width for x in x_base]
        ax.bar(positions, means, width=width, yerr=stds, capsize=4, label=setting)

    ax.set_xticks([x + width * (len(settings) - 1) / 2 for x in x_base])
    ax.set_xticklabels(tasks)
    ax.set_ylabel(metric.replace("eval_", "").replace("_", " ").title())
    ax.set_title(title)
    ax.legend(title="Setting")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot gespeichert: {out_path}")


def plot_learning_curves(history_df: pd.DataFrame, out_dir: Path):
    """Eine Lernkurven-Grafik (eval_f1 über Epochen) pro Task, eine Linie pro Setting (über Seeds gemittelt)."""
    if history_df.empty:
        print("Keine epoch_evaluation_history Daten gefunden, überspringe Lernkurven.")
        return

    for task in sorted(history_df["task"].unique()):
        sub = history_df[history_df["task"] == task]
        fig, ax = plt.subplots(figsize=(6, 4))

        for setting in sorted(sub["setting"].unique()):
            curve = (
                sub[sub["setting"] == setting]
                .groupby("epoch")["eval_f1"]
                .mean()
                .sort_index()
            )
            ax.plot(curve.index, curve.values, marker="o", label=setting)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Eval Macro F1")
        ax.set_title(f"Lernkurven – {task}")
        ax.legend(title="Setting")
        ax.set_ylim(0, 1.0)
        fig.tight_layout()

        out_path = out_dir / "plots" / f"{task}_learning_curves.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Plot gespeichert: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregiert experiment_results.json Dateien zu Tabellen und Diagrammen.")
    parser.add_argument("--results-dir", type=str, default="../results", help="Ordner mit den Unterordnern je Bedingung (Standard: results)")
    parser.add_argument("--out-dir", type=str, default=None, help="Zielordner für Tabellen/Diagramme (Standard: gleich wie --results-dir)")
    parser.add_argument("--synthetic-log", type=str, default="../data/synthetic_generation_log.json", help="Pfad zur Synthetische Generierungs-Logdatei")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir

    # Load synthetic generation log
    synthetic_log = load_synthetic_log(args.synthetic_log)

    # Flatten and save synthetic log as CSV and Markdown
    totals_df, tasks_df = flatten_synthetic_log(synthetic_log)
    totals_df.to_csv(out_dir / "synthetic_generation_log_summary.csv", index=False)
    tasks_df.to_csv(out_dir / "synthetic_generation_log_tasks.csv", index=False)
    # Markdown-Tabelle (gut zum Copy-Paste fürs Paper)
    md_path = out_dir / "synthetic_generation_log.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("## Synthetische Datengenerierung\n\n")
        f.write("### Gesamt\n\n")
        f.write(totals_df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n### Tasks\n\n")
        f.write(tasks_df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")
    print(f"Synthetische Generierungs-Logdatei gespeichert: {out_dir / 'synthetic_generation_log.csv'}")
    print(f"Synthetische Generierungs-Logdatei als Markdown gespeichert: {md_path}")

    # Load and aggregate experiment results
    summary_df, history_df = load_all_results(results_dir)
    agg_df = aggregate_across_seeds(summary_df)

    save_tables(summary_df, agg_df, out_dir)
    history_df.to_csv(out_dir / "epoch_history_long.csv", index=False)
    print(f"Epoch-Verlauf gespeichert: {out_dir / 'epoch_history_long.csv'}")

    # Plot results
    plot_bar_by_condition(agg_df, "eval_f1", out_dir / "plots" / "f1_by_condition.png", "Macro F1 je Bedingung")
    plot_bar_by_condition(agg_df, "eval_accuracy", out_dir / "plots" / "accuracy_by_condition.png", "Accuracy je Bedingung")
    plot_learning_curves(history_df, out_dir)

    print("\nFertig! Übersicht über die finalen Ergebnisse:\n")
    print("Generierung:\n")
    print(totals_df.to_string(index=False) + "\n")
    print(tasks_df.to_string(index=False) + "\n")
    print("Training:\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
