import ast
import os
import re
import sys
import string
import statistics
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
stop_words.update({'paper', 'study', 'result', 'author', 'method', 'framework',
                    'approach', 'model', 'proposed', 'using', 'use', 'based',
                    'show', 'new', 'also', 'however', 'one', 'two', 'first',
                    'data', 'performance', 'different', 'level', 'number'})

def extract_keywords(text, window_size=5, d=0.85, convergence_threshold=1e-4, max_iterations=50, max_phrase_length=3, top_n=None):
    if not isinstance(text, str) or not text.strip():
        return []

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
        return []

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

    sorted_vertices = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if top_n is None:
        top_count = max(1, len(vertices) // 3)
    else:
        top_count = min(top_n, len(vertices))
    top_words = set([word for word, score in sorted_vertices[:top_count]])

    extracted = []
    current_phrase = []
    for idx, (word, tag) in enumerate(cleaned_tokens):
        if word in top_words:
            current_phrase.append(word)
            if len(current_phrase) >= max_phrase_length:
                extracted.append(" ".join(current_phrase))
                current_phrase = []
        else:
            if current_phrase:
                if len(current_phrase) <= max_phrase_length:
                    extracted.append(" ".join(current_phrase))
                current_phrase = []
    if current_phrase and len(current_phrase) <= max_phrase_length:
        extracted.append(" ".join(current_phrase))

    return list(set(extracted))

def parse_keywords(val):
    if isinstance(val, list):
        return [str(k).lower().strip() for k in val]
    if isinstance(val, str):
        stripped = val.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return [str(k).lower().strip() for k in parsed if str(k).strip()]
            except (ValueError, SyntaxError):
                pass
        tokens = re.split(r'[;|,]', stripped)
        return [k.strip().lower() for k in tokens if k.strip()]
    return []

def stem_phrase(phrase):
    return set(stemmer.stem(w) for w in phrase.lower().strip().split() if w.strip())

def soft_match(predicted, ground_truth, jaccard_threshold=0.5):
    pred_tokens = stem_phrase(predicted)
    if not pred_tokens:
        return False
    for gt_kw in ground_truth:
        gt_tokens = stem_phrase(gt_kw)
        if not gt_tokens:
            continue
        intersection = pred_tokens & gt_tokens
        union = pred_tokens | gt_tokens
        similarity = len(intersection) / len(union) if union else 0.0
        if similarity >= jaccard_threshold:
            return True
    return False

def calculate_metrics(extracted, ground_truth, jaccard_threshold=0.5):
    extracted_clean = [e for e in extracted if e.strip()]
    gt_clean = [g for g in ground_truth if g.strip()]

    if not extracted_clean or not gt_clean:
        return 0, 0, 0

    hits = 0
    for pred in extracted_clean:
        if soft_match(pred, gt_clean, jaccard_threshold):
            hits += 1

    return hits, len(extracted_clean), len(gt_clean)

def save_to_markdown(window_size, d, threshold, max_iter, avg_p, avg_r, avg_f1):
    filepath = os.path.join(os.path.dirname(__file__), 'textrank_results.md')

    header = f"\n## Configuration: Window={window_size}, d={d}, Threshold={threshold}, MaxIter={max_iter}\n"
    content = (
        f"| Metric | Value |\n"
        f"| --- | --- |\n"
        f"| Average Precision | {avg_p:.4f} |\n"
        f"| Average Recall | {avg_r:.4f} |\n"
        f"| Average F1-Measure | {avg_f1:.4f} |\n"
    )

    mode = 'a' if os.path.exists(filepath) else 'w'
    with open(filepath, mode, encoding='utf-8') as f:
        f.write(header)
        f.write(content)

def main():
    print("loading data\n")
    df = load_parquet_data()
    print("data loaded\n")
    text_col = 'abstract' if 'abstract' in df.columns else ('text' if 'text' in df.columns else df.columns[0])
    keyword_col = 'keywords' if 'keywords' in df.columns else ('ground_truth' if 'ground_truth' in df.columns else df.columns[1])

    df = df.dropna(subset=[text_col]).reset_index(drop=True)

    window_size = 5
    d = 0.85
    threshold = 1e-4
    max_iter = 50

    total_hits = 0
    total_extracted = 0
    total_gt = 0
    results_list = []
    print("=" * 10)
    print("processing documents\n")
    print("=" * 10)

    for idx, row in df.iterrows():
        print(f"document {idx}\n")

        text = row[text_col]
        gt = parse_keywords(row[keyword_col])

        extracted = extract_keywords(
            text,
            window_size=window_size,
            d=d,
            convergence_threshold=threshold,
            max_iterations=max_iter
        )

        hits, extracted_count, gt_count = calculate_metrics(extracted, gt)
        results_list.append((hits, extracted_count, gt_count))

        total_hits += hits
        total_extracted += extracted_count
        total_gt += gt_count

    print("Evaluation Results for the First 5 Rows:")
    print("-" * 50)
    for i in range(min(5, len(df))):
        hits, extracted_count, gt_count = results_list[i]
        p = hits / extracted_count if extracted_count > 0 else 0.0
        r = hits / gt_count if gt_count > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        print(f"Row {i+1} -> Precision: {p:.4f} | Recall: {r:.4f} | F1-Measure: {f1:.4f}")
    print("-" * 50)

    num_docs = len(df)
    avg_p = total_hits / total_extracted if total_extracted > 0 else 0.0
    avg_r = total_hits / total_gt if total_gt > 0 else 0.0
    avg_f1 = (2 * avg_p * avg_r) / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0.0

    print("\nGeneral Results:")
    print(f"Total Documents Evaluated: {num_docs}")
    print(f"Micro-Averaged Precision: {avg_p:.4f}")
    print(f"Micro-Averaged Recall: {avg_r:.4f}")
    print(f"Micro-Averaged F1-Measure: {avg_f1:.4f}")

    save_to_markdown(window_size, d, threshold, max_iter, avg_p, avg_r, avg_f1)

if __name__ == "__main__":
    main()