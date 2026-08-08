import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def top_ngrams(texts, n, top_k=12, similarity_threshold=0.85, pool_multiplier=4):
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return []

    vectorizer = TfidfVectorizer(ngram_range=(n, n), stop_words="english", max_features=1000)
    matrix = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    if len(vocab) == 0:
        return []

    weights = np.asarray(matrix.sum(axis=0)).ravel()
    order = np.argsort(weights)[::-1]
    pool_size = min(len(order), top_k * pool_multiplier)
    candidate_idx = order[:pool_size]
    candidate_terms = [vocab[i] for i in candidate_idx]
    candidate_weights = [weights[i] for i in candidate_idx]

    candidate_vectors = vectorizer.transform(candidate_terms)
    sims = cosine_similarity(candidate_vectors)

    kept = []
    kept_positions = []
    for pos, (term, weight) in enumerate(zip(candidate_terms, candidate_weights)):
        duplicate = any(sims[pos, k] >= similarity_threshold for k in kept_positions)
        if duplicate:
            continue
        kept.append({"term": term, "weight": round(float(weight), 3)})
        kept_positions.append(pos)
        if len(kept) >= top_k:
            break

    return kept
