import os
import sys
from collections import Counter
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
from skopt import gp_minimize
from skopt.space import Real

nltk.download("stopwords", quiet=True)

# Define the target n-gram value for optimization
N_GRAM_SIZE = 5 

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)  # Points to src/functionalities/summary/
OUTPUT_FILE_PATH = os.path.join(SCRIPT_DIR, "ngram_optimisation.md")

# Add the parent directory to sys.path so Python can find data_loader.py
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    # Import generalized summarizer functions from the local directory
    from ngram_summarizer import (
        calculate_cosine_similarity,
        clean_and_tokenize_words,
        get_ngrams,  # Imported get_ngrams instead of get_bigrams
    )
    # Import the data loader from the parent directory
    from data_loader import load_parquet_data

except ImportError as e:
    raise ImportError(
        f"Could not import required functions. {e}\n"
        "Please ensure bigram_summarizer.py is in the BI-GRAM folder, "
        "contains the get_ngrams function, and data_loader.py is in the parent folder."
    )


def configurable_ngram_summarizer(
    document,
    n=N_GRAM_SIZE,
    num_sentences=4,
    remove_stopwords=True,
    alpha=1.0,
    pos_weight=0.0,
    redundancy_threshold=1.0,
):
    """An optimized version of the n-gram summarizer allowing parameter injection."""
    sentences = sent_tokenize(document)
    n_sent = len(sentences)
    if n_sent <= num_sentences:
        return document

    stop_words = set(stopwords.words("english"))

    doc_words = clean_and_tokenize_words(document)
    if remove_stopwords:
        doc_words = [w for w in doc_words if w not in stop_words]

    doc_ngrams = get_ngrams(doc_words, n)
    ngram_counts = Counter(doc_ngrams)
    total_ngrams = sum(ngram_counts.values())

    if total_ngrams == 0:
        return " ".join(sentences[:num_sentences])

    ngram_probs = {
        ng: count / total_ngrams for ng, count in ngram_counts.items()
    }

    sentence_scores = []
    for idx, sent in enumerate(sentences):
        sent_words = clean_and_tokenize_words(sent)
        if remove_stopwords:
            sent_words = [w for w in sent_words if w not in stop_words]
        sent_ngrams = get_ngrams(sent_words, n)

        if len(sent_ngrams) == 0:
            base_score = 0.0
        else:
            total_prob = sum(ngram_probs.get(ng, 0.0) for ng in sent_ngrams)
            base_score = total_prob / (len(sent_ngrams) ** alpha)

        is_intro_or_outro = (idx / n_sent < 0.15) or (idx / n_sent > 0.85)
        pos_multiplier = 1.0 + pos_weight if is_intro_or_outro else 1.0

        final_score = base_score * pos_multiplier
        sentence_scores.append((idx, final_score, set(sent_ngrams)))

    selected_indices = []
    selected_ngrams_accumulated = set()

    candidates = sorted(sentence_scores, key=lambda x: x[1], reverse=True)

    for idx, score, ngs in candidates:
        if len(selected_indices) >= num_sentences:
            break

        if (
            redundancy_threshold < 1.0
            and len(selected_ngrams_accumulated) > 0
            and len(ngs) > 0
        ):
            overlap = len(ngs.intersection(selected_ngrams_accumulated)) / len(
                ngs
            )
            if overlap > redundancy_threshold:
                continue

        selected_indices.append(idx)
        selected_ngrams_accumulated.update(ngs)

    if len(selected_indices) < num_sentences:
        for idx, _, _ in candidates:
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) >= num_sentences:
                break

    selected_indices.sort()
    summary_sentences = [sentences[i] for i in selected_indices]
    return " ".join(summary_sentences)


best_tracker = {"r2": -1.0}
trial_records = []


def objective(params, df_sample, scorer, num_sentences=4):
    # Unpack values suggested by the optimizer
    alpha_raw, pos_weight_raw, redundancy_threshold_raw = params

    # Apply discretization of 0.05 step inside the objective function
    alpha = round(alpha_raw / 0.05) * 0.05
    pos_weight = round(pos_weight_raw / 0.05) * 0.05
    redundancy_threshold = round(redundancy_threshold_raw / 0.05) * 0.05

    # Clip to bounds to handle float representation limits
    alpha = max(0.0, min(1.5, alpha))
    pos_weight = max(0.0, min(1.5, pos_weight))
    redundancy_threshold = max(0.2, min(1.0, redundancy_threshold))

    r1_list, r2_list, rl_list, cos_list = [], [], [], []

    for _, row in df_sample.iterrows():
        doc = row["document"]
        ref = row["abstract"]

        summary = configurable_ngram_summarizer(
            doc,
            n=N_GRAM_SIZE,
            num_sentences=num_sentences,
            remove_stopwords=True,
            alpha=alpha,
            pos_weight=pos_weight,
            redundancy_threshold=redundancy_threshold,
        )

        scores = scorer.score(ref, summary)
        r1_list.append(scores["rouge1"].fmeasure)
        r2_list.append(scores["rouge2"].fmeasure)
        rl_list.append(scores["rougeL"].fmeasure)
        cos_list.append(calculate_cosine_similarity(summary, ref))

    mean_r1 = np.mean(r1_list)
    mean_r2 = np.mean(r2_list)
    mean_rl = np.mean(rl_list)
    mean_cosine = np.mean(cos_list)

    trial_num = len(trial_records)
    
    # Track the metadata for later reporting
    record = {
        "trial_num": trial_num,
        "alpha": alpha,
        "pos_weight": pos_weight,
        "redundancy_threshold": redundancy_threshold,
        "mean_r2": mean_r2,
        "mean_r1": mean_r1,
        "mean_rl": mean_rl,
        "mean_cosine": mean_cosine,
    }
    trial_records.append(record)

    is_new_best = False
    old_best_r2 = best_tracker["r2"]
    if mean_r2 > old_best_r2:
        best_tracker["r2"] = mean_r2
        is_new_best = True

    log_line = (
        f"Trial {trial_num:03d} | alpha={alpha:.2f}, pos_weight={pos_weight:.2f}, redundancy={redundancy_threshold:.2f} | "
        f"ROUGE-1: {mean_r1:.4f}, ROUGE-2: {mean_r2:.4f}, Cosine: {mean_cosine:.4f}"
    )

    if is_new_best:
        log_line += f"  <-- ★ New Best Score! (Improved from {max(0.0, old_best_r2):.4f} to {mean_r2:.4f})"

    print(log_line)

    # To maximize ROUGE-2, return the negative value.
    return -mean_r2


def write_skopt_results_to_markdown(records, output_file=OUTPUT_FILE_PATH):
    """Parses trial data from the list of records and saves them to a Markdown file."""
    print(f"\nSaving optimization records directly to: {output_file}")

    if not records:
        print("No trial records found to save.")
        return

    # Identify best record according to highest ROUGE-2 score
    best_record = max(records, key=lambda x: x["mean_r2"])

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# scikit-optimize Hyperparameter Optimization Study (n={N_GRAM_SIZE})\n\n")
        f.write(
            f"This optimization study utilizes Bayesian optimization (Gaussian Process minimize) "
            f"to search parameters specifically configured for **{N_GRAM_SIZE}-grams**. "
            "Suggested coordinates are rounded to steps of `0.05` inside the objective.\n\n"
        )

        f.write("## Overall Best Configuration (Optimized on ROUGE-2)\n\n")
        f.write(f"* **Trial Number:** {best_record['trial_num']}\n")
        f.write(f"* **Mean ROUGE-2 F1:** {best_record['mean_r2']:.4f}\n")
        f.write(f"* **Mean ROUGE-1 F1:** {best_record['mean_r1']:.4f}\n")
        f.write(f"* **Mean ROUGE-L F1:** {best_record['mean_rl']:.4f}\n")
        f.write(f"* **Mean Cosine Similarity:** {best_record['mean_cosine']:.4f}\n\n")

        f.write("### Optimal Parameters:\n")
        f.write(f"* `alpha` (Length Normalization): **{best_record['alpha']:.2f}**\n")
        f.write(f"* `pos_weight` (MCBA Positional Bias): **{best_record['pos_weight']:.2f}**\n")
        f.write(
            f"* `redundancy_threshold` (Max Overlap): **{best_record['redundancy_threshold']:.2f}**\n\n"
        )

        f.write("## Detailed Experimentation Records (All Trials)\n\n")
        f.write(
            "| Trial ID | Alpha (Length Norm) | Pos Weight (MCBA) | Redundancy Threshold | Mean ROUGE-2 (Objective) | Mean ROUGE-1 | Mean ROUGE-L | Mean Cosine Sim |\n"
        )
        f.write(
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        )

        for t in records:
            f.write(
                f"| {t['trial_num']} | {t['alpha']:.2f} | {t['pos_weight']:.2f} | {t['redundancy_threshold']:.2f} | "
                f"{t['mean_r2']:.4f} | {t['mean_r1']:.4f} | {t['mean_rl']:.4f} | {t['mean_cosine']:.4f} |\n"
            )

    print("Markdown file generation completed.")


if __name__ == "__main__":
    df = load_parquet_data()
    sample_df = df.head(5)

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    # Define hyperparameter search spaces
    space = [
        Real(0.0, 1.5, name="alpha"),
        Real(0.0, 1.5, name="pos_weight"),
        Real(0.2, 1.0, name="redundancy_threshold"),
    ]

    n_trials = 150
    print(
        f"Starting scikit-optimize search across {n_trials} trials for {N_GRAM_SIZE}-grams using rounded step sizes of 0.05...\n"
    )

    # Perform minimization search (we target negative ROUGE-2 to achieve maximization)
    gp_minimize(
        func=lambda params: objective(params, sample_df, scorer, num_sentences=4),
        dimensions=space,
        n_calls=n_trials,
        random_state=42,  # set seed for reproducible initialization steps
    )

    write_skopt_results_to_markdown(trial_records)