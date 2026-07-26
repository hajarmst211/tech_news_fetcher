import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MD_FILE = os.path.join(SCRIPT_DIR, "../bigram_optimisation.md")
VIS_DIR = os.path.join(SCRIPT_DIR)
os.makedirs(VIS_DIR, exist_ok=True)

BEST_ALPHA = 0.60
BEST_POS_WEIGHT = 0.00
BEST_REDUNDANCY = 0.20


def parse_markdown_table(filepath):
    rows = []
    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line.startswith("|") or line.startswith("|---") or line.startswith("| Trial"):
            continue
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) < 8:
            continue
        try:
            rows.append({
                "trial": int(parts[0]),
                "alpha": float(parts[1]),
                "pos_weight": float(parts[2]),
                "redundancy_threshold": float(parts[3]),
                "Mean ROUGE-2": float(parts[4]),
                "Mean ROUGE-1": float(parts[5]),
                "Mean ROUGE-L": float(parts[6]),
                "Mean Cosine Sim": float(parts[7]),
            })
        except (ValueError, IndexError):
            continue
    return pd.DataFrame(rows)


def plot_violin_distributions(df):
    hp_configs = [
        ("alpha", "Alpha (Length Norm)"),
        ("pos_weight", "Pos Weight (MCBA)"),
        ("redundancy_threshold", "Redundancy Threshold (Max Overlap)"),
    ]
    y_col = "Mean ROUGE-2"

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True, constrained_layout=True)
    fig.suptitle(
        "Violin Plots: Distribution of Mean ROUGE-2 per Hyperparameter Value",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for idx, (hp_key, hp_label) in enumerate(hp_configs):
        ax = axes[idx]
        sns.violinplot(
            x=hp_key, y=y_col, data=df, ax=ax,
            inner="quartile", hue=hp_key, palette="viridis", cut=0,
            linewidth=1, alpha=0.7, legend=False,
        )

        best_mask = (
            (df["alpha"] == BEST_ALPHA)
            & (df["pos_weight"] == BEST_POS_WEIGHT)
            & (df["redundancy_threshold"] == BEST_REDUNDANCY)
        )
        best_row = df[best_mask].iloc[0] if best_mask.any() else df.loc[df[y_col].idxmax()]
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

        if hp_key == "alpha":
            ax.set_ylabel("Mean ROUGE-2", fontsize=11)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=9)

    out = os.path.join(VIS_DIR, "plot1_violin_distributions.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot 1 saved to: {out}")


def plot_interaction_heatmap(df):
    target_redundancy = 0.20
    subset = df[df["redundancy_threshold"] == target_redundancy].copy()

    if subset.empty:
        print(f"No trials found with redundancy_threshold == {target_redundancy}. Skipping heatmap.")
        return

    pivot = subset.pivot_table(
        index="alpha", columns="pos_weight",
        values="Mean ROUGE-2", aggfunc="mean",
    )

    pivot = pivot.sort_index(ascending=True)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(
        f"Interaction Heatmap: Alpha vs Pos Weight\n"
        f"(Slice at Redundancy Threshold = {target_redundancy:.2f}, "
        f"color = Mean ROUGE-2)",
        fontsize=13, fontweight="bold",
    )

    mask_missing = pivot.isna()

    sns.heatmap(
        pivot, annot=True, fmt=".4f", linewidths=0.8, linecolor="white",
        cmap="viridis", ax=ax, mask=mask_missing,
        annot_kws={"size": 8, "fontweight": "bold"},
        cbar_kws={"label": "Mean ROUGE-2"},
    )

    if mask_missing.any().any():
        missing_mask = mask_missing.values
        for i in range(missing_mask.shape[0]):
            for j in range(missing_mask.shape[1]):
                if missing_mask[i, j]:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True,
                                               facecolor="#D3D3D3", edgecolor="white", linewidth=0.8))
                    ax.text(j + 0.5, i + 0.5, "N/A", ha="center", va="center",
                            fontsize=7, color="#555555", fontstyle="italic")

    best_mask = (
        (subset["alpha"] == BEST_ALPHA)
        & (subset["pos_weight"] == BEST_POS_WEIGHT)
    )
    if best_mask.any():
        best_alpha_idx = list(pivot.index).index(BEST_ALPHA) if BEST_ALPHA in pivot.index else None
        best_pw_idx = list(pivot.columns).index(BEST_POS_WEIGHT) if BEST_POS_WEIGHT in pivot.columns else None
        if best_alpha_idx is not None and best_pw_idx is not None:
            ax.add_patch(plt.Rectangle(
                (best_pw_idx, best_alpha_idx), 1, 1, fill=False,
                edgecolor="red", linewidth=3, linestyle="--",
            ))
            ax.text(
                best_pw_idx + 0.5, best_alpha_idx - 0.4, "BEST",
                ha="center", va="bottom", fontsize=8, color="red", fontweight="bold",
            )

    ax.set_xlabel("Pos Weight (MCBA)", fontsize=11)
    ax.set_ylabel("Alpha (Length Norm)", fontsize=11)
    ax.tick_params(axis="both", labelsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(VIS_DIR, "plot2_interaction_heatmap.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot 2 saved to: {out}")




def main():
    df = parse_markdown_table(MD_FILE)
    if df.empty:
        print("No data found. Run optimize_bigrma.py first.")
        return

    print(f"Parsed {len(df)} trials.")
    print(f"Best trial: alpha={BEST_ALPHA}, pos_weight={BEST_POS_WEIGHT}, "
          f"redundancy={BEST_REDUNDANCY} -> ROUGE-2={df['Mean ROUGE-2'].max():.4f}\n")

    plot_violin_distributions(df)
    plot_interaction_heatmap(df)
    print("\nAll visualizations generated successfully.")


if __name__ == "__main__":
    main()
