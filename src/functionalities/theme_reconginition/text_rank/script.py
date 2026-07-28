import os
import sys
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader import load_parquet_data, load_data

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'cs_papers_api.csv')

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

print("Loading sentence-transformer model...")
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

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


def save_to_markdown(window_size, d, threshold, max_iter, sim_threshold, avg_p, avg_r, avg_f1, avg_similarity, doc_details):
    filepath = os.path.join(os.path.dirname(__file__), 'textrank_results.md')
    
    header = (
        f"\n## Configuration: Window={window_size}, d={d}, Threshold={threshold}, "
        f"MaxIter={max_iter}, Semantic Match Threshold={sim_threshold}\n\n"
    )
    
    table_details = (
        "| Document Index | Actual Label (Dataset) | Predicted Topic (Model) | Semantic Similarity | Match? |\n"
        "| --- | --- | --- | --- | --- |\n"
    )
    for idx, actual, predicted, score, match in doc_details:
        match_str = "Yes" if match else "No"
        table_details += f"| {idx} | {actual} | {predicted} | {score:.4f} | {match_str} |\n"
        
    metrics_summary = (
        f"\n### Performance Metrics\n"
        f"| Metric | Value |\n"
        f"| --- | --- |\n"
        f"| Average Semantic Similarity | {avg_similarity:.4f} |\n"
        f"| Average Precision (on threshold) | {avg_p:.4f} |\n"
        f"| Average Recall (on threshold) | {avg_r:.4f} |\n"
        f"| Average F1-Measure (on threshold) | {avg_f1:.4f} |\n"
    )

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write(table_details)
        f.write(metrics_summary)


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    label_col = 'label' if 'label' in df.columns else df.columns[1]

    df = df.dropna(subset=[text_col]).reset_index(drop=True)

    window_size = 5
    d = 0.85
    threshold = 1e-4
    max_iter = 50
    semantic_match_threshold = 0.45

    predicted_topics = []
    actual_labels = []

    print(f"Extracting topics from {len(df)} documents...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="TextRank Extraction"):
        text = row[text_col]
        actual_label = str(row[label_col]).strip()
        
        predicted_topic = extract_single_topic(
            text,
            window_size=window_size,
            d=d,
            convergence_threshold=threshold,
            max_iterations=max_iter
        )
        predicted_topics.append(predicted_topic)
        actual_labels.append(actual_label)

    print("Encoding predicted topics and actual labels in batches...")

    clean_preds = [p if p else " " for p in predicted_topics]
    clean_actuals = [a if a else " " for a in actual_labels]

    pred_embeddings = similarity_model.encode(clean_preds, convert_to_tensor=True, show_progress_bar=True)
    actual_embeddings = similarity_model.encode(clean_actuals, convert_to_tensor=True, show_progress_bar=True)

    print("Evaluating semantic similarity...")
    pred_norms = torch.nn.functional.normalize(pred_embeddings, p=2, dim=1)
    actual_norms = torch.nn.functional.normalize(actual_embeddings, p=2, dim=1)
    similarities = (pred_norms * actual_norms).sum(dim=1).tolist()

    total_hits = 0
    total_extracted = 0
    total_gt = 0
    total_similarity = 0.0
    doc_details = []

    for idx in range(len(df)):
        pred = predicted_topics[idx]
        actual = actual_labels[idx]
        score = similarities[idx]
        is_match = score >= semantic_match_threshold

        doc_details.append((idx, actual, pred, score, is_match))

        if is_match:
            total_hits += 1
        if pred:
            total_extracted += 1
        if actual:
            total_gt += 1
        total_similarity += score

    num_docs = len(df)
    avg_similarity = total_similarity / num_docs if num_docs > 0 else 0.0
    avg_p = total_hits / total_extracted if total_extracted > 0 else 0.0
    avg_r = total_hits / total_gt if total_gt > 0 else 0.0
    avg_f1 = (2 * avg_p * avg_r) / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0.0

    print("Saving results to markdown...")
    save_to_markdown(
        window_size, d, threshold, max_iter, 
        semantic_match_threshold, avg_p, avg_r, avg_f1, 
        avg_similarity, doc_details
    )
    print("Done!")


if __name__ == "__main__":
    main()