import os
import re
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
    scores = [
        ("mean_std", "Mean Std (Multi-Seed)"),
        ("mean_mcba", "Mean MCBA (Multi-Seed)"),
        ("mean_rpm", "Mean RPM (Multi-Seed)"),
        ("overall_mean", "Overall Mean"),
    ]

    fig, axes = plt.subplots(len(hyperparams), len(scores), figsize=(20, 24))
    fig.suptitle("Optuna Hyperparameter Optimization Results", fontsize=16, y=0.98)

    for row, (hp_key, hp_label) in enumerate(hyperparams):
        for col, (sc_key, sc_label) in enumerate(scores):
            ax = axes[row][col]
            hp_vals = [d[hp_key] for d in data]
            sc_vals = [d[sc_key] for d in data]

            ax.scatter(hp_vals, sc_vals, alpha=0.6, edgecolors="k", linewidths=0.5, s=40)

            # Trend line
            if len(set(hp_vals)) > 1:
                z = np.polyfit(hp_vals, sc_vals, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(hp_vals), max(hp_vals), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.7, linewidth=1.5)

            if col == 0:
                ax.set_ylabel(hp_label, fontsize=10)
            else:
                ax.set_ylabel("")
            if row == len(hyperparams) - 1:
                ax.set_xlabel(sc_label, fontsize=9)
            else:
                ax.set_xlabel("")
            ax.tick_params(axis="both", labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
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
