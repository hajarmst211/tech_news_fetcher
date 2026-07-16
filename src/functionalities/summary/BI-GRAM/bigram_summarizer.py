import os
import re
import urllib.request
from collections import Counter
import nltk
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize


def download_and_load_data():
    """Downloads and loads the specified arXiv dataset partition."""
    local_path = "train-00000-of-00015.parquet"
    if not os.path.exists(local_path):
        url = "https://huggingface.co/datasets/ccdv/arxiv-summarization/resolve/main/document/train-00000-of-00015.parquet"
        print("Local file not found. Starting dataset download...")
        urllib.request.urlretrieve(url, local_path)
        print("\nDownload finished.")

    print("Loading data from local storage...")
    df = pd.read_parquet(local_path, columns=["article", "abstract"])
    df = df.rename(columns={"article": "document"})
    return df


def clean_and_tokenize_words(text):
    """Tokenizes text into lowercase words, removing non-alphabetic tokens."""
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if re.match(r"^[a-z]+$", word)]


def get_bigrams(words):
    """Generates a list of bigrams from a list of words."""
    return [(words[i], words[i + 1]) for i in range(len(words) - 1)]


def bigram_frequency_summarizer(document, num_sentences=3):
    """Summarizes a document by scoring sentences based on document-wide bigram frequencies."""
    sentences = sent_tokenize(document)
    if len(sentences) <= num_sentences:
        return document

    doc_words = clean_and_tokenize_words(document)
    doc_bigrams = get_bigrams(doc_words)
    bigram_counts = Counter(doc_bigrams)

    if not bigram_counts:
        return " ".join(sentences[:num_sentences])

    total_bigrams = sum(bigram_counts.values())
    bigram_probs = {bg: count / total_bigrams for bg, count in bigram_counts.items()}

    sentence_scores = []
    for idx, sent in enumerate(sentences):
        sent_words = clean_and_tokenize_words(sent)
        sent_bigrams = get_bigrams(sent_words)

        if len(sent_bigrams) == 0:
            score = 0.0
        else:
            total_score = sum(bigram_probs.get(bg, 0.0) for bg in sent_bigrams)
            score = total_score / len(sent_bigrams)

        sentence_scores.append((idx, score))

    top_sentence_indices = sorted(
        sentence_scores, key=lambda x: x[1], reverse=True
    )[:num_sentences]
    top_sentence_indices.sort(key=lambda x: x[0])

    summary_sentences = [sentences[idx] for idx, _ in top_sentence_indices]
    return " ".join(summary_sentences)


# ==========================================
# Evaluation Functions
# ==========================================


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


def evaluate_summaries(df_sample, num_sentences=4):
    """Generates summaries and evaluates them against reference abstracts."""
    # Initialize ROUGE Scorer
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    r1_scores, r2_scores, rl_scores = [], [], []
    cosine_scores = []

    print(
        f"\nStarting evaluation over {len(df_sample)} documents (extracting {num_sentences} sentences per summary)..."
    )

    for idx, row in df_sample.iterrows():
        doc = row["document"]
        ref = row["abstract"]

        # Generate summary
        gen_sum = bigram_frequency_summarizer(doc, num_sentences=num_sentences)

        # Calculate ROUGE
        scores = scorer.score(ref, gen_sum)
        r1_scores.append(scores["rouge1"].fmeasure)
        r2_scores.append(scores["rouge2"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)

        # Calculate Cosine Similarity
        cos_sim = calculate_cosine_similarity(gen_sum, ref)
        cosine_scores.append(cos_sim)

    # Compile average metrics
    metrics = {
        "Mean ROUGE-1 (F1)": np.mean(r1_scores),
        "Mean ROUGE-2 (F1)": np.mean(r2_scores),
        "Mean ROUGE-L (F1)": np.mean(rl_scores),
        "Mean Cosine Similarity": np.mean(cosine_scores),
    }

    return metrics


# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    df = download_and_load_data()

    # Evaluate on a sample batch (e.g., first 10 documents for quick computation)
    batch_size = 10
    sample_df = df.head(batch_size)

    # Perform evaluation
    results = evaluate_summaries(sample_df, num_sentences=4)

    # Output Results
    print("\n================ Evaluation Results ================")
    for metric, score in results.items():
        print(f"{metric:<25}: {score:.4f}")
    print("====================================================")