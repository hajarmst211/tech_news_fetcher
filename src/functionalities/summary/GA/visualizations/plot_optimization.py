import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MD_FILE = os.path.join(SCRIPT_DIR, "../optimisation_output.md")
VIS_DIR = os.path.join(SCRIPT_DIR)
os.makedirs(VIS_DIR, exist_ok=True)


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
                "Mean Std": float(parts[7]),
                "Mean MCBA": float(parts[8]),
                "Mean RPM": float(parts[9]),
                "Overall Mean": float(parts[10]),
            })
        except (ValueError, IndexError):
            continue
    return pd.DataFrame(rows)


def plot_violin_distributions(df):
    y_col = "Overall Mean"

    hp_configs = [
        ("p_crossover", "P Crossover"),
        ("p_mutation", "P Mutation"),
        ("w_position_mcba", "W Pos MCBA"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True, constrained_layout=True)
    fig.suptitle(
        "Violin Plots: Distribution of Overall Mean F1 per Hyperparameter Value",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for idx, (hp_key, hp_label) in enumerate(hp_configs):
        ax = axes[idx]
        sns.violinplot(
            x=hp_key, y=y_col, data=df, ax=ax,
            inner="quartile", hue=hp_key, palette="viridis", cut=0,
            linewidth=1, alpha=0.7, legend=False,
        )

        best_idx = df[y_col].idxmax()
        best_row = df.loc[best_idx]
        best_x = list(ax.get_xticks()).index(
            ax.get_xticks()[[str(t.get_text()) for t in ax.get_xticklabels()].index(str(best_row[hp_key]))]
        ) if str(best_row[hp_key]) in [str(t.get_text()) for t in ax.get_xticklabels()] else 0
        ax.scatter(
            best_x, best_row[y_col],
            marker="*", s=350, c="red", edgecolors="black",
            linewidths=1.2, zorder=5, label="Best Trial",
        )

        ax.set_xlabel(hp_label, fontsize=11)
        ax.set_title(hp_label, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        ax.tick_params(axis="both", labelsize=9)

        if hp_key == "p_crossover":
            ax.set_ylabel("Overall Mean F1", fontsize=11)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=9)

    out = os.path.join(VIS_DIR, "plot1_violin_distributions.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot 1 saved to: {out}")


def plot_interaction_heatmap(df):
    target_coverage = 0.50
    subset = df[df["w_coverage"] == target_coverage].copy()

    if subset.empty:
        print(f"No trials found with w_coverage == {target_coverage}. Skipping heatmap.")
        return

    pivot = subset.pivot_table(
        index="p_crossover", columns="w_position_mcba",
        values="Overall Mean", aggfunc="mean",
    )

    pivot = pivot.sort_index(ascending=True)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(
        f"Interaction Heatmap: P Crossover vs W Pos MCBA\n"
        f"(Slice where W Coverage is fixed at 0.50, "
        f"color = Overall Mean F1)",
        fontsize=13, fontweight="bold",
    )

    mask_missing = pivot.isna()

    sns.heatmap(
        pivot, annot=True, fmt=".4f", linewidths=0.8, linecolor="white",
        cmap="viridis", ax=ax, mask=mask_missing,
        annot_kws={"size": 9, "fontweight": "bold"},
        cbar_kws={"label": "Overall Mean F1"},
    )

    if mask_missing.any().any():
        for i in range(mask_missing.shape[0]):
            for j in range(mask_missing.shape[1]):
                if mask_missing.iloc[i, j]:
                    ax.add_patch(plt.Rectangle(
                        (j, i), 1, 1, fill=True,
                        facecolor="#D3D3D3", edgecolor="white", linewidth=0.8,
                    ))
                    ax.text(j + 0.5, i + 0.5, "N/A", ha="center", va="center",
                            fontsize=7, color="#555555", fontstyle="italic")

    best_idx = df["Overall Mean"].idxmax()
    best_row = df.loc[best_idx]
    if (best_row["w_coverage"] == target_coverage
            and best_row["p_crossover"] in pivot.index
            and best_row["w_position_mcba"] in pivot.columns):
        r = list(pivot.index).index(best_row["p_crossover"])
        c = list(pivot.columns).index(best_row["w_position_mcba"])
        ax.add_patch(plt.Rectangle(
            (c, r), 1, 1, fill=False,
            edgecolor="red", linewidth=3, linestyle="--",
        ))
        ax.text(c + 0.5, r - 0.4, "BEST", ha="center", va="bottom",
                fontsize=8, color="red", fontweight="bold")

    ax.set_xlabel("W Pos MCBA", fontsize=11)
    ax.set_ylabel("P Crossover", fontsize=11)
    ax.tick_params(axis="both", labelsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(VIS_DIR, "plot2_interaction_heatmap.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot 2 saved to: {out}")




def main():
    df = parse_markdown_table(MD_FILE)
    if df.empty:
        print("No data found. Run optimize_ga.py first.")
        return

    print(f"Parsed {len(df)} trials.")
    best_idx = df["Overall Mean"].idxmax()
    best_row = df.loc[best_idx]
    print(f"Best trial {best_row['trial']}: pop_size={best_row['pop_size']}, "
          f"p_crossover={best_row['p_crossover']}, p_mutation={best_row['p_mutation']}, "
          f"w_coverage={best_row['w_coverage']}, w_pos_mcba={best_row['w_position_mcba']} "
          f"-> Overall Mean={best_row['Overall Mean']:.4f}\n")

    plot_violin_distributions(df)
    plot_interaction_heatmap(df)
    print("\nAll visualizations generated successfully.")


if __name__ == "__main__":
    main()
