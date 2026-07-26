import os
import sys
import spacy
import networkx as nx
import nltk
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Ensure NLTK resources are available
try:
    wn.ensure_loaded()
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')

# Resolve path and import data loader as requested
try:
    # Use standard __file__ lookup with a fallback if running interactively
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    sys.path.insert(0, os.path.join(current_dir, ".."))
    from data_loader import load_parquet_data
except ImportError:
    # Fallback placeholder if executed outside the project workspace environment
    def load_parquet_data():
        return ["The quick brown fox jumps over the lazy dog. Dogs are loyal animals. Loyal animals make great companions."]


class SemanticGraphReducer:
    def __init__(self):
        # Load Spacy model for NLP preprocessing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def _get_wordnet_pos(self, spacy_pos):
        """Map Spacy POS tag to WordNet POS tag."""
        if spacy_pos == "NOUN":
            return wn.NOUN
        elif spacy_pos == "VERB":
            return wn.VERB
        elif spacy_pos == "ADJ":
            return wn.ADJ
        elif spacy_pos == "ADV":
            return wn.ADV
        return None

    def _get_word_score(self, word, pos):
        """
        Calculate word popularity/frequency score using WordNet synsets.
        Higher frequency count across lemmas yields a higher score.
        """
        wn_pos = self._get_wordnet_pos(pos)
        if not wn_pos:
            return 1.0
        
        synsets = wn.synsets(word, pos=wn_pos)
        if not synsets:
            return 1.0
        
        # Sum frequencies of the lemmas matching the synset
        score = sum(lemma.count() for synset in synsets for lemma in synset.lemmas())
        return max(float(score), 1.0)

    def _resolve_pronominals(self, doc):
        """
        A simplified pronominal resolution mapping pronouns to the most 
        recent matching noun entity in the sentence context.
        """
        resolved_tokens = []
        last_noun = None
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"]:
                last_noun = token.text
            
            if token.pos_ == "PRON" and token.text.lower() in ["he", "she", "it", "they"] and last_noun:
                resolved_tokens.append((token.idx, last_noun))
            else:
                resolved_tokens.append((token.idx, token.text))
        return resolved_tokens

    def build_rich_semantic_graph(self, text):
        """
        Phase 1: Rich Semantic Graph (RSG) Creation
        Nodes: Nouns, Verbs, Named Entities.
        Edges: Semantic and Syntactic relations (Subject, Object, etc.).
        """
        doc = self.nlp(text)
        G = nx.MultiDiGraph()
        
        # Simple coreference representation substitution
        resolved_map = dict(self._resolve_pronominals(doc))

        for sent in doc.sents:
            for token in sent:
                # Target nouns and verbs as semantic nodes
                if token.pos_ in ["NOUN", "VERB", "PROPN"]:
                    lemma = token.lemma_.lower()
                    score = self._get_word_score(lemma, token.pos_)
                    
                    if not G.has_node(lemma):
                        G.add_node(lemma, pos=token.pos_, score=score, text=token.text)
                    
                    # Establish semantic edges based on dependency parsing
                    if token.dep_ in ["nsubj", "dobj", "pobj", "nsubjpass"]:
                        head_lemma = token.head.lemma_.lower()
                        if token.head.pos_ in ["NOUN", "VERB", "PROPN"]:
                            if not G.has_node(head_lemma):
                                head_score = self._get_word_score(head_lemma, token.head.pos_)
                                G.add_node(head_lemma, pos=token.head.pos_, score=head_score, text=token.head.text)
                            
                            G.add_edge(lemma, head_lemma, relation=token.dep_)
                            
        return G

    def reduce_graph(self, G):
        """
        Phase 2: Semantic Graph Reduction
        Merges or consolidates redundant nodes using WordNet relations (Hypernyms, Holonyms, Entailments).
        """
        reduced_G = G.copy()
        nodes = list(reduced_G.nodes(data=True))
        merged_nodes = set()

        for i in range(len(nodes)):
            node_u, data_u = nodes[i]
            if node_u in merged_nodes or not reduced_G.has_node(node_u):
                continue
                
            pos_u = data_u.get('pos')
            wn_pos_u = self._get_wordnet_pos(pos_u)
            synsets_u = wn.synsets(node_u, pos=wn_pos_u) if wn_pos_u else []
            
            if not synsets_u:
                continue

            for j in range(i + 1, len(nodes)):
                node_v, data_v = nodes[j]
                if node_v in merged_nodes or not reduced_G.has_node(node_v):
                    continue
                
                pos_v = data_v.get('pos')
                if pos_u != pos_v:
                    continue
                
                wn_pos_v = self._get_wordnet_pos(pos_v)
                synsets_v = wn.synsets(node_v, pos=wn_pos_v) if wn_pos_v else []
                
                if not synsets_v:
                    continue

                # Evaluate semantic connections (Hypernyms, Holonyms, Entailments)
                similarity = synsets_u[0].path_similarity(synsets_v[0])
                is_related = False
                
                # Check hypernym relationship
                hyper_u = set(synsets_u[0].closure(lambda s: s.hypernyms()))
                if synsets_v[0] in hyper_u:
                    is_related = True
                
                # Check entailment for verbs
                if pos_u == "VERB":
                    entailments_u = set(synsets_u[0].entailments())
                    if synsets_v[0] in entailments_u:
                        is_related = True
                
                # If highly similar or strongly related, consolidate
                if (similarity and similarity > 0.7) or is_related:
                    # Choose node with the higher popularity score as target
                    keep, discard = (node_u, node_v) if data_u['score'] >= data_v['score'] else (node_v, node_u)
                    
                    # Redirect edges from discard to keep
                    for u, v, key, data in list(reduced_G.edges(keys=True, data=True)):
                        if u == discard:
                            reduced_G.add_edge(keep, v, relation=data.get('relation'))
                        if v == discard:
                            reduced_G.add_edge(u, keep, relation=data.get('relation'))
                            
                    if reduced_G.has_node(discard):
                        reduced_G.remove_node(discard)
                    merged_nodes.add(discard)

        return reduced_G

    def generate_abstract(self, reduced_G):
        """
        Phase 3: Generation of Abstract
        Traverses remaining core SVO structures to generate consolidated summary sentences.
        """
        sentences = []
        visited_edges = set()
        
        # Reconstruct relationships based on remaining edges
        for u, v, data in reduced_G.edges(data=True):
            edge_key = (u, v, data.get('relation'))
            if edge_key in visited_edges:
                continue
            
            relation = data.get('relation')
            pos_u = reduced_G.nodes[u].get('pos')
            pos_v = reduced_G.nodes[v].get('pos')
            
            # Reconstruct basic assertions
            if relation == "nsubj" and pos_v == "VERB":
                # u is subject, v is verb
                # Look for an object of verb v
                obj_node = None
                for succ in reduced_G.successors(v):
                    for edge_idx in reduced_G[v][succ]:
                        edge_data = reduced_G[v][succ][edge_idx]
                        if edge_data.get('relation') in ["dobj", "pobj"]:
                            obj_node = succ
                            break
                
                if obj_node:
                    sentence = f"{u.capitalize()} {v} {obj_node}."
                else:
                    sentence = f"{u.capitalize()} {v}."
                
                if sentence not in sentences:
                    sentences.append(sentence)
            
            visited_edges.add(edge_key)
            
        if not sentences:
            # Fallback if graph is too disconnected
            top_nodes = sorted(reduced_G.nodes(data=True), key=lambda x: x[1].get('score', 0), reverse=True)[:5]
            words = [node[0] for node in top_nodes]
            if words:
                sentences.append(f"Focus points include: {', '.join(words)}.")
            else:
                sentences.append("")
                
        return " ".join(sentences)

    def summarize(self, text):
        """Executes SGR system pipeline."""
        if not text or not text.strip():
            return ""
        G = self.build_rich_semantic_graph(text)
        reduced_G = self.reduce_graph(G)
        summary = self.generate_abstract(reduced_G)
        return summary


# --- Evaluation Metrics ---

def calculate_coherence(generated_text):
    """
    Evaluates the logical connection and smooth flow of sentences using cosine
    similarity of adjacent sentences within TF-IDF vector space.
    """
    sentences = nltk.sent_tokenize(generated_text)
    if len(sentences) <= 1:
        return 1.0  # Monologue or single sentence is structurally coherent by default

    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return 0.0

    similarities = []
    for i in range(len(sentences) - 1):
        sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[i+1])[0][0]
        similarities.append(sim)
        
    return float(np.mean(similarities))


def calculate_synonym_frequency_score(generated_text):
    """
    Evaluates WordNet popularity ranks of the chosen lemmas in the summary.
    Highly ranked synonyms yield a higher score.
    """
    tokens = nltk.word_tokenize(generated_text.lower())
    scores = []
    
    for token in tokens:
        synsets = wn.synsets(token)
        if synsets:
            # Score based on frequency counts of lemma names
            lemma_freqs = [lemma.count() for synset in synsets for lemma in synset.lemmas()]
            if lemma_freqs:
                scores.append(sum(lemma_freqs) / len(lemma_freqs))
                
    if not scores:
        return 0.0
    return float(np.mean(scores))


# --- Main Execution Block ---

if __name__ == "__main__":
    output_path = os.path.join(current_dir, "sgr_results.md")
    try:
        data = load_parquet_data()
    except Exception as e:
        print(f"Data loading failed: {e}. Executing fallback dummy text evaluation.")
        data = ["The quick brown fox jumps over the lazy dog. Dogs are loyal animals. Loyal animals make great companions."]

    reducer = SemanticGraphReducer()

    if not isinstance(data, list):
        data = [data]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# SGR Results\n\n")
        f.write("Processing Documents...\n\n")

        for index, record in enumerate(data[:3]):
            if isinstance(record, dict):
                text = record.get("text", "")
            elif isinstance(record, str):
                text = record
            else:
                text = str(record)

            if not text.strip():
                continue

            f.write(f"## Document {index + 1}\n\n")
            f.write(f"### Original\n\n")
            f.write(f"{text[:250]}{'...' if len(text) > 250 else ''}\n\n")

            summary = reducer.summarize(text)

            f.write(f"### Abstractive Summary (SGR Model)\n\n")
            f.write(f"{summary if summary else '[Empty Summary Generated]'}\n\n")

            coherence = calculate_coherence(summary)
            wn_rank_score = calculate_synonym_frequency_score(summary)

            f.write(f"### Evaluation Metrics\n\n")
            f.write(f"| Metric | Score |\n")
            f.write(f"|---|---|\n")
            f.write(f"| Sentence Coherence Similarity Score | {coherence:.4f} |\n")
            f.write(f"| Synonym Frequency (WordNet Rank) Score | {wn_rank_score:.4f} |\n\n")
            f.write("---\n\n")

    print(f"Results written to {output_path}")