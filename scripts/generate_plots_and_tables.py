# ! To be used in external folder, where the used synthetic data (`data`) and `results` of the different dataset sizes have been copied to in a folder named `results` into subfolders named `{sample_size_per_class}_seeds-{seed1-seed2-...}`

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Konfiguration ---
INPUT_DIR = "results"  # Ordner mit den Experimentergebnissen
OUTPUT_DIR = "latex_figures"  # Ordner für die generierten Diagramme/Tabellen
TASKS = ["SST2", "SNLI"]  # Aufgaben
SETTINGS = ["Human", "Synthetic", "Mixed"]  # Settings
METRICS = ["eval_accuracy", "eval_f1"]  # Metriken (aus all_results.csv)
SAMPLE_SIZES = [10, 50, 100, 500, 1000]  # Erwartete Datensatzgrößen pro Klasse

# --- Ordner erstellen ---
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# --- Daten einlesen und nach sample_size gruppieren ---
def load_and_aggregate_results():
    """Lädt alle CSV-Dateien aus den Unterordnern und aggregiert sie nach sample_size_per_class."""
    all_data = []

    # Durchsuche alle Unterordner in INPUT_DIR
    for folder in os.listdir(INPUT_DIR):
        folder_path = os.path.join(INPUT_DIR, folder) #TODO: , INPUT_DIR?

        # Extrahiere sample_size_per_class aus dem Ordnernamen (z. B. "1000_seeds-42-123-7" -> 1000)
        try:
            sample_size = int(folder.split("_")[0])
        except (ValueError, IndexError):
            continue  # Überspringe Ordner, die nicht dem Muster entsprechen

        if sample_size not in SAMPLE_SIZES:
            continue  # Überspringe unerwartete Größen

        # Lade alle all_results.csv-Dateien in diesem Ordner
        csv_files = glob.glob(os.path.join(folder_path, "*", "all_results.csv"))
        if not csv_files:
            csv_files = glob.glob(os.path.join(folder_path, "all_results.csv"))

        print(folder_path, csv_files)
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)

            # Prüfe, ob die CSV-Datei die benötigten Spalten hat
            required_columns = ["task", "setting", "seed"] + METRICS
            if not all(col in df.columns for col in required_columns):
                print(f"⚠️ Überspringe {csv_file}: Fehlende Spalten {set(required_columns) - set(df.columns)}")
                continue

            # Füge sample_size_per_class als Spalte hinzu
            df["sample_size_per_class"] = sample_size

            # Filtere nur die relevanten Spalten
            df = df[["task", "setting", "sample_size_per_class", "seed"] + METRICS]
            all_data.append(df)

    print(f"✅ Alle Daten geladen: {len(all_data)} CSV-Dateien gefunden.")
    if not all_data:
        print("❌ Keine passenden CSV-Dateien gefunden. Überprüfe INPUT_DIR und die Spaltennamen.")
        return pd.DataFrame()

    # Kombiniere alle Daten
    combined_df = pd.concat(all_data, ignore_index=True)

    # Aggregiere nach task, setting, sample_size_per_class
    aggregated_df = combined_df.groupby(["task", "setting", "sample_size_per_class"]).agg(
        {metric: ["mean", "std"] for metric in METRICS}
    ).reset_index()

    # Flache Spaltennamen (z. B. "eval_accuracy_mean" statt ("eval_accuracy", "mean"))
    aggregated_df.columns = [
        f"{col[0]}_{col[1]}" if col[1] in ["mean", "std"] else col[0]
        for col in aggregated_df.columns
    ]

    print("aggregated_df:", aggregated_df)

    return aggregated_df

# --- Tabellen für LaTeX generieren ---
def generate_latex_tables(aggregated_df):
    """Generiert LaTeX-Tabellen für die Performance nach Datensatzgröße."""
    for task in TASKS:
        task_df = aggregated_df[aggregated_df["task"] == task]
        if task_df.empty:
            continue

        latex_table = f"\\begin{{table}}[h]\n\\centering\n\\caption{{Performance on {task} by dataset size per class (Mean $\\pm$ Std).}}\n\\label{{tab:{task.lower()}_scaling}}\n\\begin{{tabular}}{{lccc}}\n\\toprule\n"
        latex_table += f"\\textbf{{Size}} & \\textbf{{Setting}} & \\textbf{{Accuracy}} & \\textbf{{Macro F1}} \\\\ \\midrule\n"

        for size in SAMPLE_SIZES:
            size_df = task_df[task_df["sample_size_per_class"] == size]
            if size_df.empty:
                continue

            for setting in SETTINGS:
                setting_df = size_df[size_df["setting"] == setting]
                if setting_df.empty:
                    continue

                accuracy_mean = setting_df["eval_accuracy_mean"].values[0]
                accuracy_std = setting_df["eval_accuracy_std"].values[0]
                f1_mean = setting_df["eval_f1_mean"].values[0]
                f1_std = setting_df["eval_f1_std"].values[0]

                # Füge $...$ um ± hinzu
                latex_table += f"{size} & {setting} & {accuracy_mean:.3f} $\\pm$ {accuracy_std:.3f} & {f1_mean:.3f} $\\pm$ {f1_std:.3f} \\\\ "

            latex_table += "\\midrule\n"

        latex_table += "\\bottomrule\n\\end{tabular}\n\\end{table}\n\n"
        with open(f"{OUTPUT_DIR}/{task.lower()}_table.tex", "w", encoding="utf-8") as f:
            f.write(latex_table)
        print(f"✅ LaTeX-Tabelle für {task} generiert: {OUTPUT_DIR}/{task.lower()}_table.tex")

# --- Diagramme generieren ---
def generate_plots(aggregated_df):
    """Generiert Linienplots für Accuracy und Macro F1 vs. Datensatzgröße."""
    for metric in METRICS:
        metric_name = metric.replace("eval_", "").title()
        for task in TASKS:
            task_df = aggregated_df[aggregated_df["task"] == task]
            print("task_df:", task_df)
            if task_df.empty:
                continue

            plt.figure(figsize=(10, 6))

            for setting in SETTINGS:
                setting_df = task_df[task_df["setting"] == setting]
                print("setting_df:", setting_df)
                if setting_df.empty:
                    continue

                sizes = setting_df["sample_size_per_class"].values
                means = setting_df[f"{metric}_mean"].values
                stds = setting_df[f"{metric}_std"].values

                print("sizes:", sizes)
                print("means:", means)
                print("stds:", stds)

                plt.errorbar(
                    sizes,
                    means,
                    yerr=stds,
                    fmt='o-',
                    capsize=5,
                    label=setting,
                    linewidth=2,
                    markersize=8
                )

            # plt.xscale('log')
            plt.xlabel('Dataset Size per Class')
            plt.ylabel(metric_name)
            plt.title(f'{task}: {metric_name} vs. Dataset Size per Class')
            plt.legend()
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.tight_layout()

            # Speichern als PDF und PNG
            output_path = f"{OUTPUT_DIR}/{task.lower()}_{metric}"
            plt.savefig(f"{output_path}.pdf", bbox_inches='tight', dpi=300)
            plt.savefig(f"{output_path}.png", bbox_inches='tight', dpi=300)
            plt.close()

            print(f"✅ Diagramm generiert: {output_path}.pdf")

# --- Hauptfunktion ---
def main():
    print("🔍 Lade und aggregiere Ergebnisse...")
    aggregated_df = load_and_aggregate_results()

    if aggregated_df.empty:
        print("❌ Keine Daten gefunden. Überprüfe INPUT_DIR und die Spaltennamen.")
        return

    print("\n📊 Generiere LaTeX-Tabellen...")
    generate_latex_tables(aggregated_df)

    print("\n📈 Generiere Diagramme...")
    generate_plots(aggregated_df)

    print(f"\n✅ Fertig! Ergebnisse sind in '{OUTPUT_DIR}' gespeichert.")

if __name__ == "__main__":
    main()