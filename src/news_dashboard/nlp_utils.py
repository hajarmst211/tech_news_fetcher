import re
import nltk
import numpy as np
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

_model = None

try:
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
except Exception:
    pass

CUSTOM_STOP_WORDS = {
    "https", "http", "com", "net", "org", "www", "github", "huggingface",
    "submit", "tracked", "packages", "site", "web", "html", "using"
}

ALL_STOP_WORDS = list(text.ENGLISH_STOP_WORDS.union(CUSTOM_STOP_WORDS))


def pre_clean_text(text_str):
    if not isinstance(text_str, str):
        return ""

    text_str = text_str.lower()
    text_str = re.sub(r"https?://\S+|www\.\S+", "", text_str)
    text_str = re.sub(r"\b\S+\.(com|net|org|dev|io|html|co|us|sh|py)\b", "", text_str)
    text_str = re.sub(r"[^a-z\s]", " ", text_str)
    text_str = re.sub(r"\s+", " ", text_str).strip()
    return text_str


def is_grammatically_meaningful(term, n):
    words = term.split()
    if len(words) != n:
        return False

    if any(len(w) < 2 for w in words):
        return False

    try:
        tagged = nltk.pos_tag(words)
    except Exception:
        return True

    tags = [tag for word, tag in tagged]

    if tags[0] in ("IN", "CC", "DT", "PRP", "PRP$", "MD", "TO"):
        return False

    if n > 1:
        if tags[0] in ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "RB", "RBR", "RBS"):
            return False

    has_noun = any(t.startswith("NN") for t in tags)
    if not has_noun:
        return False

    return True


def top_ngrams(texts, n, top_k=12, similarity_threshold=0.85, pool_multiplier=4):
    cleaned_texts = [pre_clean_text(t) for t in texts]
    cleaned_texts = [t for t in cleaned_texts if t.strip()]
    if not cleaned_texts:
        return []

    vectorizer = TfidfVectorizer(ngram_range=(n, n), stop_words=ALL_STOP_WORDS, max_features=1000)
    matrix = vectorizer.fit_transform(cleaned_texts)
    vocab = vectorizer.get_feature_names_out()
    if len(vocab) == 0:
        return []

    weights = np.asarray(matrix.sum(axis=0)).ravel()
    order = np.argsort(weights)[::-1]

    pool_size = min(len(order), top_k * pool_multiplier * 2)
    candidate_idx = order[:pool_size]

    raw_candidate_terms = [vocab[i] for i in candidate_idx]
    raw_candidate_weights = [weights[i] for i in candidate_idx]

    candidate_terms = []
    candidate_weights = []
    for term, weight in zip(raw_candidate_terms, raw_candidate_weights):
        if is_grammatically_meaningful(term, n):
            candidate_terms.append(term)
            candidate_weights.append(weight)

    if not candidate_terms:
        return []

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


def get_sentence_transformer_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def cluster_topics(raw_topics, threshold=0.55):

    if not raw_topics:
        return {}, {}
        
    model = get_sentence_transformer_model()
    embeddings = model.encode(raw_topics, convert_to_tensor=True)
    
    assigned = set()
    canonical_to_raw = {}
    raw_to_canonical = {}
    
    for idx, topic in enumerate(raw_topics):
        if topic in assigned:
            continue
            
        cos_scores = util.cos_sim(embeddings[idx], embeddings)[0]
        
        similar_indices = [i for i, score in enumerate(cos_scores) if score >= threshold]
        
        cluster_members = []
        for i in similar_indices:
            member = raw_topics[i]
            if member not in assigned:
                cluster_members.append(member)
                assigned.add(member)
                
        if cluster_members:
            canonical = min(cluster_members, key=len)
            canonical_to_raw[canonical] = cluster_members
            for member in cluster_members:
                raw_to_canonical[member] = canonical
                
    return canonical_to_raw, raw_to_canonical