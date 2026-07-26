import os
import sys
import re
from collections import Counter
import nltk
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup paths to load dataset
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data_loader import load_parquet_data

# Download NLTK resources
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize


def clean_and_tokenize_words(text):
    """Tokenizes text into lowercase words, removing non-alphabetic tokens."""
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if re.match(r"^[a-z]+$", word)]


def get_ngrams(words, n):
    """Generates a list of n-grams from a list of words."""
    if len(words) < n:
        return []
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def ngram_frequency_summarizer(document, n=2, num_sentences=3):
    """Summarizes a document by scoring sentences based on document-wide n-gram frequencies."""
    sentences = sent_tokenize(document)
    if len(sentences) <= num_sentences:
        return document

    doc_words = clean_and_tokenize_words(document)
    doc_ngrams = get_ngrams(doc_words, n)
    ngram_counts = Counter(doc_ngrams)

    if not ngram_counts:
        return " ".join(sentences[:num_sentences])

    total_ngrams = sum(ngram_counts.values())
    ngram_probs = {ng: count / total_ngrams for ng, count in ngram_counts.items()}

    sentence_scores = []
    for idx, sent in enumerate(sentences):
        sent_words = clean_and_tokenize_words(sent)
        sent_ngrams = get_ngrams(sent_words, n)

        if len(sent_ngrams) == 0:
            score = 0.0
        else:
            total_score = sum(ngram_probs.get(ng, 0.0) for ng in sent_ngrams)
            score = total_score / len(sent_ngrams)

        sentence_scores.append((idx, score))

    top_sentence_indices = sorted(
        sentence_scores, key=lambda x: x[1], reverse=True
    )[:num_sentences]
    top_sentence_indices.sort(key=lambda x: x[0])

    summary_sentences = [sentences[idx] for idx, _ in top_sentence_indices]
    return " ".join(summary_sentences)


# Evaluation

def calculate_cosine_similarity(text1, text2):
    """Calculates TF-IDF cosine similarity between two texts."""
    if not text1.strip() or not text2.strip():
        return 0.0
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return sim[0][0]
    except ValueError:
        # Handles edge cases where no terms can be vectorized
        return 0.0


def evaluate_summaries_for_n(df_sample, n, num_sentences=4):
    """Generates summaries using n-grams and evaluates them against reference abstracts."""
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    r1_scores, r2_scores, rl_scores = [], [], []
    cosine_scores = []

    for idx, row in df_sample.iterrows():
        doc = row["document"]
        ref = row["abstract"]

        # Generate summary using specific n-gram length
        gen_sum = ngram_frequency_summarizer(doc, n=n, num_sentences=num_sentences)

        scores = scorer.score(ref, gen_sum)
        r1_scores.append(scores["rouge1"].fmeasure)
        r2_scores.append(scores["rouge2"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)

        cos_sim = calculate_cosine_similarity(gen_sum, ref)
        cosine_scores.append(cos_sim)

    metrics = {
        "n": n,
        "Mean ROUGE-1 (F1)": np.mean(r1_scores),
        "Mean ROUGE-2 (F1)": np.mean(r2_scores),
        "Mean ROUGE-L (F1)": np.mean(rl_scores),
        "Mean Cosine Similarity": np.mean(cosine_scores),
    }

    return metrics


def write_results_to_markdown(results_list, output_filepath):
    """Writes the evaluation outcomes to a structured markdown file."""
    # Find the row with the highest ROUGE-L score to identify an optimal parameter for this sample
    best_n_row = max(results_list, key=lambda x: x["Mean ROUGE-L (F1)"])
    best_n = best_n_row["n"]

    md_content = []
    md_content.append("# N-Gram Length Comparison Report\n")
    md_content.append(
        "This report evaluates the performance of the statistical extractive summarization model "
        "using different n-gram sizes (from unigrams to 5-grams). The evaluation is measured against "
        "the provided dataset abstracts.\n"
    )

    # Table Header
    md_content.append("| n (Gram Size) | Mean ROUGE-1 (F1) | Mean ROUGE-2 (F1) | Mean ROUGE-L (F1) | Mean Cosine Similarity |")
    md_content.append("| :--- | :--- | :--- | :--- | :--- |")

    # Table Rows
    for r in results_list:
        md_content.append(
            f"| **{r['n']}** | {r['Mean ROUGE-1 (F1)']:.4f} | {r['Mean ROUGE-2 (F1)']:.4f} | {r['Mean ROUGE-L (F1)']:.4f} | {r['Mean Cosine Similarity']:.4f} |"
        )

    md_content.append("\n## Observations\n")
    md_content.append(
        f"- For this specific evaluation batch, an n-gram size of **n={best_n}** "
        f"yielded the highest Mean ROUGE-L score ({best_n_row['Mean ROUGE-L (F1)']:.4f}).\n"
        "- Lower values of n (e.g., 1 or 2) score based on more general word frequencies, while larger values "
        "of n (such as 4 or 5) match longer contiguous phrases, which can sometimes lead to lower scores due to "
        "fewer exact matches in shorter documents."
    )

    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))


if __name__ == "__main__":
    df = load_parquet_data()

    # Define sample batch size
    batch_size = 10
    sample_df = df.head(batch_size)

    # Test values of n from 1 to 5
    n_values = [1, 2, 3, 4, 5]
    results = []

    print(f"Starting evaluation across n-values {n_values} on {batch_size} documents...")
    for n in n_values:
        print(f"Evaluating n={n}...")
        metrics = evaluate_summaries_for_n(sample_df, n=n, num_sentences=4)
        results.append(metrics)

    # Save results to markdown file in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "n_gram_test.md")
    
    write_results_to_markdown(results, output_file)
    print(f"\nEvaluation completed. Results written to: {output_file}")