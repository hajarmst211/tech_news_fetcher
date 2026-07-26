import os
import sys
import string
import itertools
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

def extract_keywords(text, window_size=5, d=0.85, convergence_threshold=0.0001, max_iterations=50, max_phrase_length=3, top_n=15):
    tokens = word_tokenize(str(text))
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
    top_count = min(top_n, len(vertices))
    top_words = set([word for word, score in sorted_vertices[:top_count]])
    
    punct_set = set(string.punctuation)
    extracted = []
    current_phrase = []
    for idx, (word, tag) in enumerate(cleaned_tokens):
        if word in top_words:
            if current_phrase and idx > 0:
                prev_token = tokens[idx - 1] if idx - 1 < len(tokens) else ''
                if prev_token in punct_set:
                    if len(current_phrase) <= max_phrase_length:
                        extracted.append(" ".join(current_phrase))
                    current_phrase = []
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
        for sep in [';', ',', '|']:
            if sep in val:
                return [k.lower().strip() for k in val.split(sep) if k.strip()]
        return [val.lower().strip()]
    return []

def calculate_metrics(extracted, ground_truth):
    def stem_phrase(phrase):
        return " ".join(stemmer.stem(w) for w in phrase.lower().strip().split())
    
    extracted_set = set(stem_phrase(e) for e in extracted if e.strip())
    gt_set = set(stem_phrase(g) for g in ground_truth if g.strip())
    
    if not extracted_set or not gt_set:
        return 0.0, 0.0, 0.0
        
    hits = len(extracted_set.intersection(gt_set))
    precision = hits / len(extracted_set)
    recall = hits / len(gt_set)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
        
    return precision, recall, f1

def run_evaluation(df, text_col, keyword_col, params):
    total_p, total_r, total_f1 = 0.0, 0.0, 0.0
    num_docs = len(df)
    for _, row in df.iterrows():
        text = row[text_col]
        gt = parse_keywords(row[keyword_col])
        extracted = extract_keywords(
            text, 
            window_size=params['window_size'], 
            d=params['d'], 
            convergence_threshold=params['convergence_threshold'],
            max_phrase_length=params['max_phrase_length'],
            top_n=params['top_n']
        )
        p, r, f1 = calculate_metrics(extracted, gt)
        total_p += p
        total_r += r
        total_f1 += f1
    avg_p = total_p / num_docs if num_docs > 0 else 0.0
    avg_r = total_r / num_docs if num_docs > 0 else 0.0
    avg_f1 = total_f1 / num_docs if num_docs > 0 else 0.0
    return avg_p, avg_r, avg_f1

def save_results_to_markdown(df_results, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# Grid Search Evaluation Results\n\n")
        f.write("| Rank | Window Size | d | Threshold | Max Phrase Length | Top N | Avg Precision | Avg Recall | Avg F1-Score |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for rank, (_, row) in enumerate(df_results.iterrows(), 1):
            f.write(f"| {rank} | {int(row['window_size'])} | {row['d']:.2f} | {row['convergence_threshold']} | {int(row['max_phrase_length'])} | {int(row['top_n'])} | {row['avg_precision']:.4f} | {row['avg_recall']:.4f} | {row['avg_f1']:.4f} |\n")

def main():
    df = load_parquet_data()
    sample_size = min(50, len(df))
    df_sample = df.sample(n=sample_size, random_state=42).copy()
    text_col = 'abstract' if 'abstract' in df.columns else ('text' if 'text' in df.columns else df.columns[0])
    keyword_col = 'keywords' if 'keywords' in df.columns else ('ground_truth' if 'ground_truth' in df.columns else df.columns[1])
    
    grid_params = {
        'window_size': [5],
        'd': [0.75, 0.78, 0.80, 0.83, 0.85],
        'convergence_threshold': [0.1, 0.15, 0.2, 0.25, 0.3],
        'max_phrase_length': [2, 3, 4, 5],
        'top_n': [5, 10]
    }
    
    keys, values = zip(*grid_params.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    results = []
    
    for idx, params in enumerate(experiments):
        avg_p, avg_r, avg_f1 = run_evaluation(df_sample, text_col, keyword_col, params)
        results.append({
            'window_size': params['window_size'],
            'd': params['d'],
            'convergence_threshold': params['convergence_threshold'],
            'max_phrase_length': params['max_phrase_length'],
            'top_n': params['top_n'],
            'avg_precision': avg_p,
            'avg_recall': avg_r,
            'avg_f1': avg_f1
        })
        
    df_results = pd.DataFrame(results).sort_values(by='avg_f1', ascending=False)
    output_path = os.path.join(os.path.dirname(__file__), 'grid_search_results.md')
    save_results_to_markdown(df_results, output_path)

if __name__ == "__main__":
    main()