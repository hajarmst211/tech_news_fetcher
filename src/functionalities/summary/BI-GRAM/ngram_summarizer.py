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


def evaluate_summaries(df_sample, n=2, num_sentences=4):
    """Generates summaries and evaluates them against reference abstracts using n-grams."""
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    r1_scores, r2_scores, rl_scores = [], [], []
    cosine_scores = []

    print(
        f"\nStarting evaluation over {len(df_sample)} documents (extracting {num_sentences} sentences per summary, using n={n})..."
    )

    for idx, row in df_sample.iterrows():
        doc = row["document"]
        ref = row["abstract"]

        # Generate summary using the configured n-gram setting
        gen_sum = ngram_frequency_summarizer(doc, n=n, num_sentences=num_sentences)

        scores = scorer.score(ref, gen_sum)
        r1_scores.append(scores["rouge1"].fmeasure)
        r2_scores.append(scores["rouge2"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)

        cos_sim = calculate_cosine_similarity(gen_sum, ref)
        cosine_scores.append(cos_sim)

    metrics = {
        "Mean ROUGE-1 (F1)": np.mean(r1_scores),
        "Mean ROUGE-2 (F1)": np.mean(r2_scores),
        "Mean ROUGE-L (F1)": np.mean(rl_scores),
        "Mean Cosine Similarity": np.mean(cosine_scores),
    }

    return metrics


if __name__ == "__main__":
    df = load_parquet_data()

    # Evaluate on a sample batch (e.g., first 10 documents for quick computation)
    batch_size = 10
    sample_df = df.head(batch_size)

    # Perform evaluation with a chosen value of n (e.g., n=2 for bigrams, n=3 for trigrams)
    chosen_n = 2
    results = evaluate_summaries(sample_df, n=chosen_n, num_sentences=4)

    # Output Results
    print(f"\n================ Evaluation Results (n={chosen_n}) ================")
    for metric, score in results.items():
        print(f"{metric:<25}: {score:.4f}")
    print("====================================================")