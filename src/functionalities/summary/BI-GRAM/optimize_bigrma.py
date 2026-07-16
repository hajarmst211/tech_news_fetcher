import os
import optuna
from collections import Counter
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer

# Download NLTK resources if not present
nltk.download("stopwords", quiet=True)

# Determine the exact directory where this script is saved
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE_PATH = os.path.join(SCRIPT_DIR, "bigram_optimisation.md")

# Import baseline functions from bigram_summarizer.py
try:
    from bigram_summarizer import (
        calculate_cosine_similarity,
        clean_and_tokenize_words,
        download_and_load_data,
        get_bigrams,
    )
except ImportError:
    raise ImportError(
        "Could not import functions from bigram_summarizer.py. "
        "Please ensure both files are in the same folder."
    )


def configurable_bigram_summarizer(
    document,
    num_sentences=4,
    remove_stopwords=True,  # Always True
    alpha=1.0,
    pos_weight=0.0,
    redundancy_threshold=1.0,
):
    """An optimized version of the bigram summarizer allowing parameter injection."""
    sentences = sent_tokenize(document)
    n_sent = len(sentences)
    if n_sent <= num_sentences:
        return document

    stop_words = set(stopwords.words("english"))

    # Create document bigram probability map
    doc_words = clean_and_tokenize_words(document)
    doc_words = [w for w in doc_words if w not in stop_words]

    doc_bigrams = get_bigrams(doc_words)
    bigram_counts = Counter(doc_bigrams)
    total_bigrams = sum(bigram_counts.values())

    if total_bigrams == 0:
        return " ".join(sentences[:num_sentences])

    bigram_probs = {
        bg: count / total_bigrams for bg, count in bigram_counts.items()
    }

    # Score each sentence
    sentence_scores = []
    for idx, sent in enumerate(sentences):
        sent_words = clean_and_tokenize_words(sent)
        sent_words = [w for w in sent_words if w not in stop_words]
        sent_bigrams = get_bigrams(sent_words)

        if len(sent_bigrams) == 0:
            base_score = 0.0
        else:
            total_prob = sum(bigram_probs.get(bg, 0.0) for bg in sent_bigrams)
            # Apply length normalization
            base_score = total_prob / (len(sent_bigrams) ** alpha)

        # Apply MCBA-style position bias (first 15% or last 15% of the text)
        is_intro_or_outro = (idx / n_sent < 0.15) or (idx / n_sent > 0.85)
        pos_multiplier = 1.0 + pos_weight if is_intro_or_outro else 1.0

        final_score = base_score * pos_multiplier
        sentence_scores.append((idx, final_score, set(sent_bigrams)))

    # Selection phase with redundancy filtering (MMR approximation)
    selected_indices = []
    selected_bigrams_accumulated = set()

    candidates = sorted(sentence_scores, key=lambda x: x[1], reverse=True)

    for idx, score, bgs in candidates:
        if len(selected_indices) >= num_sentences:
            break

        # Check redundancy constraints
        if (
            redundancy_threshold < 1.0
            and len(selected_bigrams_accumulated) > 0
            and len(bgs) > 0
        ):
            overlap = len(bgs.intersection(selected_bigrams_accumulated)) / len(
                bgs
            )
            if overlap > redundancy_threshold:
                continue

        selected_indices.append(idx)
        selected_bigrams_accumulated.update(bgs)

    # Fallback if selection was too strict
    if len(selected_indices) < num_sentences:
        for idx, _, _ in candidates:
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) >= num_sentences:
                break

    # Reassemble summary chronologically
    selected_indices.sort()
    summary_sentences = [sentences[i] for i in selected_indices]
    return " ".join(summary_sentences)


# ==========================================
# Optuna Study Objective with Tracking State
# ==========================================

# Global dictionary to track the overall best score across all trials
best_tracker = {"r2": -1.0}


def objective(trial, df_sample, scorer, num_sentences=4):
    # Suggest parameters with fine-grained step resolutions (0.05 step size)
    alpha = trial.suggest_float("alpha", 0.0, 1.5, step=0.05)
    pos_weight = trial.suggest_float("pos_weight", 0.0, 1.5, step=0.05)
    redundancy_threshold = trial.suggest_float(
        "redundancy_threshold", 0.2, 1.0, step=0.05
    )

    r1_list, r2_list, rl_list, cos_list = [], [], [], []

    for _, row in df_sample.iterrows():
        doc = row["document"]
        ref = row["abstract"]

        # Generate summary
        summary = configurable_bigram_summarizer(
            doc,
            num_sentences=num_sentences,
            remove_stopwords=True,
            alpha=alpha,
            pos_weight=pos_weight,
            redundancy_threshold=redundancy_threshold,
        )

        # Calculate scores
        scores = scorer.score(ref, summary)
        r1_list.append(scores["rouge1"].fmeasure)
        r2_list.append(scores["rouge2"].fmeasure)
        rl_list.append(scores["rougeL"].fmeasure)
        cos_list.append(calculate_cosine_similarity(summary, ref))

    mean_r2 = np.mean(r2_list)

    # Store additional performance metrics inside Optuna's trial attributes
    trial.set_user_attr("mean_r1", np.mean(r1_list))
    trial.set_user_attr("mean_rl", np.mean(rl_list))
    trial.set_user_attr("mean_cosine", np.mean(cos_list))

    # Check if this trial beat the previous best score
    is_new_best = False
    old_best_r2 = best_tracker["r2"]
    if mean_r2 > old_best_r2:
        best_tracker["r2"] = mean_r2
        is_new_best = True

    # Build the terminal log line
    log_line = (
        f"Trial {trial.number:03d} | alpha={alpha:.2f}, pos_weight={pos_weight:.2f}, redundancy={redundancy_threshold:.2f} | "
        f"ROUGE-1: {np.mean(r1_list):.4f}, ROUGE-2: {mean_r2:.4f}, Cosine: {np.mean(cos_list):.4f}"
    )

    if is_new_best:
        log_line += f"  <-- ★ New Best Score! (Improved from {max(0.0, old_best_r2):.4f} to {mean_r2:.4f})"

    print(log_line)

    return mean_r2


def write_optuna_results_to_markdown(study, output_file=OUTPUT_FILE_PATH):
    """Parses trial data from the Optuna study and saves them to a Markdown file."""
    print(f"\nSaving optimization records directly to: {output_file}")

    trials = study.trials
    completed_trials = [
        t for t in trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    best_trial = study.best_trial

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Optuna Hyperparameter Optimization Study\n\n")
        f.write(
            "This optimization study utilizes Bayesian optimization (TPE Sampler) "
            "to continuously search the parameters using narrow, fine-grained steps (step size: `0.05`).\n\n"
        )

        f.write("## Overall Best Configuration (Optimized on ROUGE-2)\n\n")
        f.write(f"* **Trial Number:** {best_trial.number}\n")
        f.write(f"* **Mean ROUGE-2 F1:** {best_trial.value:.4f}\n")
        f.write(
            f"* **Mean ROUGE-1 F1:** {best_trial.user_attrs.get('mean_r1', 0.0):.4f}\n"
        )
        f.write(
            f"* **Mean ROUGE-L F1:** {best_trial.user_attrs.get('mean_rl', 0.0):.4f}\n"
        )
        f.write(
            f"* **Mean Cosine Similarity:** {best_trial.user_attrs.get('mean_cosine', 0.0):.4f}\n\n"
        )

        f.write("### Optimal Parameters:\n")
        f.write(f"* `alpha` (Length Normalization): **{best_trial.params['alpha']:.2f}**\n")
        f.write(f"* `pos_weight` (MCBA Positional Bias): **{best_trial.params['pos_weight']:.2f}**\n")
        f.write(
            f"* `redundancy_threshold` (Max Overlap): **{best_trial.params['redundancy_threshold']:.2f}**\n\n"
        )

        f.write("## Detailed Experimentation Records (All Trials)\n\n")
        f.write(
            "| Trial ID | Alpha (Length Norm) | Pos Weight (MCBA) | Redundancy Threshold | Mean ROUGE-2 (Objective) | Mean ROUGE-1 | Mean ROUGE-L | Mean Cosine Sim |\n"
        )
        f.write(
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        )

        for t in completed_trials:
            f.write(
                f"| {t.number} | {t.params['alpha']:.2f} | {t.params['pos_weight']:.2f} | {t.params['redundancy_threshold']:.2f} | "
                f"{t.value:.4f} | {t.user_attrs.get('mean_r1', 0.0):.4f} | {t.user_attrs.get('mean_rl', 0.0):.4f} | {t.user_attrs.get('mean_cosine', 0.0):.4f} |\n"
            )

    print("Markdown file generation completed.")


# ==========================================
# Run Optimization
# ==========================================
if __name__ == "__main__":
    df = download_and_load_data()
    sample_df = df.head(5)

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    # Disable default Optuna logs so we only show our detailed prints
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Initialize the Bayesian Optimization Study
    study = optuna.create_study(direction="maximize")

    n_trials = 150
    print(
        f"Starting Optuna search across {n_trials} trials using step sizes of 0.05...\n"
    )

    study.optimize(
        lambda trial: objective(trial, sample_df, scorer, num_sentences=4),
        n_trials=n_trials,
    )

    # Save output to bigram_optimisation.md in the current folder of the script
    write_optuna_results_to_markdown(study)