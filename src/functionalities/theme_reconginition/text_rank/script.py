import ast
import os
import re
import sys
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords, wordnet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader import load_parquet_data

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def is_valid_pos(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']

def penn_to_wn(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    return None

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def extract_single_topic(text, window_size=5, d=0.85, convergence_threshold=1e-4, max_iterations=50, max_phrase_length=3):
    if not isinstance(text, str) or not text.strip():
        return ""

    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    cleaned_tokens = []
    for word, tag in tagged:
        wn_tag = penn_to_wn(tag)
        if wn_tag:
            lemma = lemmatizer.lemmatize(word.lower(), pos=wn_tag)
        else:
            lemma = word.lower()
        cleaned_tokens.append((lemma, tag))

    vertices = set()
    for word, tag in cleaned_tokens:
        if is_valid_pos(tag) and word not in stop_words:
            vertices.add(word)

    if not vertices:
        return ""

    graph = {v: set() for v in vertices}
    n = len(cleaned_tokens)
    for i in range(n):
        word_i, tag_i = cleaned_tokens[i]
        if not is_valid_pos(tag_i) or word_i in stop_words:
            continue
        for j in range(i + 1, min(i + window_size, n)):
            word_j, tag_j = cleaned_tokens[j]
            if is_valid_pos(tag_j) and word_j not in stop_words and word_i != word_j:
                graph[word_i].add(word_j)
                graph[word_j].add(word_i)

    scores = {v: 1.0 for v in vertices}

    for _ in range(max_iterations):
        new_scores = {}
        max_diff = 0
        for node in vertices:
            incoming_sum = 0
            for neighbor in graph[node]:
                out_degree = len(graph[neighbor])
                if out_degree > 0:
                    incoming_sum += scores[neighbor] / out_degree
            new_score = (1 - d) + d * incoming_sum
            new_scores[node] = new_score
            max_diff = max(max_diff, abs(new_score - scores[node]))
        scores = new_scores
        if max_diff < convergence_threshold:
            break

    top_count = max(1, len(vertices) // 3)
    sorted_vertices = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_words = set([word for word, score in sorted_vertices[:top_count]])

    candidates = []
    current_phrase = []
    for word, tag in cleaned_tokens:
        if word in top_words:
            current_phrase.append(word)
            if len(current_phrase) >= max_phrase_length:
                candidates.append(" ".join(current_phrase))
                current_phrase = []
        else:
            if current_phrase:
                candidates.append(" ".join(current_phrase))
                current_phrase = []
    if current_phrase:
        candidates.append(" ".join(current_phrase))

    unique_candidates = list(set(candidates))
    if not unique_candidates:
        return ""

    candidate_scores = {}
    for cand in unique_candidates:
        cand_words = cand.split()
        candidate_scores[cand] = sum(scores.get(w, 0.0) for w in cand_words)

    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_candidates[0][0]

def stem_phrase(phrase):
    return set(stemmer.stem(w) for w in phrase.lower().strip().split() if w.strip())

def soft_match(predicted, ground_truth, jaccard_threshold=0.3):
    pred_tokens = stem_phrase(predicted)
    gt_tokens = stem_phrase(ground_truth)
    if not pred_tokens or not gt_tokens:
        return False
    intersection = pred_tokens & gt_tokens
    union = pred_tokens | gt_tokens
    similarity = len(intersection) / len(union) if union else 0.0
    return similarity >= jaccard_threshold

def save_to_markdown(window_size, d, threshold, max_iter, avg_p, avg_r, avg_f1, doc_details):
    filepath = os.path.join(os.path.dirname(__file__), 'textrank_results.md')
    
    header = f"\n## Configuration: Window={window_size}, d={d}, Threshold={threshold}, MaxIter={max_iter}\n\n"
    
    table_details = "| Document Index | Actual Label (Dataset) | Predicted Topic (Model) | Match? |\n| --- | --- | --- | --- |\n"
    for idx, actual, predicted, match in doc_details:
        match_str = "Yes" if match else "No"
        table_details += f"| {idx} | {actual} | {predicted} | {match_str} |\n"
        
    metrics_summary = (
        f"\n### Performance Metrics\n"
        f"| Metric | Value |\n"
        f"| --- | --- |\n"
        f"| Average Precision | {avg_p:.4f} |\n"
        f"| Average Recall | {avg_r:.4f} |\n"
        f"| Average F1-Measure | {avg_f1:.4f} |\n"
    )

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write(table_details)
        f.write(metrics_summary)

def main():
    df = load_parquet_data()
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    label_col = 'label' if 'label' in df.columns else df.columns[1]

    df = df.dropna(subset=[text_col]).reset_index(drop=True)

    window_size = 5
    d = 0.85
    threshold = 1e-4
    max_iter = 50

    total_hits = 0
    total_extracted = 0
    total_gt = 0
    doc_details = []

    for idx, row in df.iterrows():
        text = row[text_col]
        actual_label = str(row[label_col]).strip()

        predicted_topic = extract_single_topic(
            text,
            window_size=window_size,
            d=d,
            convergence_threshold=threshold,
            max_iterations=max_iter
        )

        is_match = soft_match(predicted_topic, actual_label)
        doc_details.append((idx, actual_label, predicted_topic, is_match))

        hits = 1 if is_match else 0
        total_hits += hits
        total_extracted += 1 if predicted_topic else 0
        total_gt += 1 if actual_label else 0

    avg_p = total_hits / total_extracted if total_extracted > 0 else 0.0
    avg_r = total_hits / total_gt if total_gt > 0 else 0.0
    avg_f1 = (2 * avg_p * avg_r) / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0.0

    save_to_markdown(window_size, d, threshold, max_iter, avg_p, avg_r, avg_f1, doc_details)

if __name__ == "__main__":
    main()