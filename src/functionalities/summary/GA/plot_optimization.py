import os
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MD_FILE = os.path.join(SCRIPT_DIR, "optimisation_output.md")
OUTPUT_PNG = os.path.join(SCRIPT_DIR, "optimization_plots.png")


def parse_markdown_table(filepath):
    rows = []
    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line.startswith("|") or line.startswith("|---") or line.startswith("| Trial"):
            continue
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) < 11:
            continue
        try:
            rows.append({
                "trial": int(parts[0]),
                "pop_size": int(parts[1]),
                "p_crossover": float(parts[2]),
                "p_mutation": float(parts[3]),
                "w_coverage": float(parts[4]),
                "w_position_mcba": float(parts[5]),
                "penalty_weight": float(parts[6]),
                "mean_std": float(parts[7]),
                "mean_mcba": float(parts[8]),
                "mean_rpm": float(parts[9]),
                "overall_mean": float(parts[10]),
            })
        except (ValueError, IndexError):
            continue
    return rows


def plot_optimization(data):
    hyperparams = [
        ("pop_size", "Pop Size"),
        ("p_crossover", "P Crossover"),
        ("p_mutation", "P Mutation"),
        ("w_coverage", "W Coverage"),
        ("w_position_mcba", "W Pos MCBA"),
        ("penalty_weight", "Penalty Weight"),
    ]
    score_key = "overall_mean"
    score_label = "Overall Mean F1"

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Overall Mean Score by Hyperparameter", fontsize=14)

    for idx, (hp_key, hp_label) in enumerate(hyperparams):
        ax = axes[idx // 3][idx % 3]
        hp_vals = [d[hp_key] for d in data]
        sc_vals = [d[score_key] for d in data]

        # Group by unique hp values and compute mean score
        unique_vals = sorted(set(hp_vals))
        means = [np.mean([sc_vals[i] for i in range(len(hp_vals)) if hp_vals[i] == v]) for v in unique_vals]

        ax.bar([str(v) for v in unique_vals], means, color="steelblue", edgecolor="black")
        ax.set_title(hp_label)
        ax.set_xlabel(hp_label)
        ax.set_ylabel(score_label)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {OUTPUT_PNG}")


if __name__ == "__main__":
    data = parse_markdown_table(MD_FILE)
    if not data:
        print("No data found. Run optimize_ga.py first.")
    else:
        print(f"Parsed {len(data)} trials.")
        plot_optimization(data)
