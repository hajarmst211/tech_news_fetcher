import os
import sys
import numpy as np
import networkx as nx
import nltk
from collections import Counter


# Import classes and evaluation metrics from the baseline reduction file
try:
    from semantic_graph_reduction import (
        SemanticGraphReducer,
        calculate_coherence,
        calculate_synonym_frequency_score,
        load_parquet_data
    )
except ImportError:
    sys.path.append(os.getcwd())
    from semantic_graph_reduction import (
        SemanticGraphReducer,
        calculate_coherence,
        calculate_synonym_frequency_score,
        load_parquet_data
    )

def calculate_unigram_f_measure(generated_text, reference_text):
    """
    Calculates the token-level Unigram F-measure (ROUGE-1 F1 approximation)
    between the generated summary and the ground-truth reference abstract.
    """
    if not generated_text.strip() or not reference_text.strip():
        return 0.0

    # Tokenize and normalize to lowercase
    gen_tokens = nltk.word_tokenize(generated_text.lower())
    ref_tokens = nltk.word_tokenize(reference_text.lower())

    # Remove punctuation tokens
    gen_tokens = [t for t in gen_tokens if t.isalnum()]
    ref_tokens = [t for t in ref_tokens if t.isalnum()]

    if not gen_tokens or not ref_tokens:
        return 0.0

    # Calculate token overlap using multisets (handles duplicate words)
    gen_counter = Counter(gen_tokens)
    ref_counter = Counter(ref_tokens)
    overlap = sum((gen_counter & ref_counter).values())

    # Precision, Recall, and F1 calculations
    precision = overlap / len(gen_tokens)
    recall = overlap / len(ref_tokens)

    if precision + recall == 0.0:
        return 0.0

    f1_score = 2 * (precision * recall) / (precision + recall)
    return float(f1_score)

class RobustSemanticGraphReducer(SemanticGraphReducer):
    """
    An enhanced abstract generator that handles passive subjects, 
    nominal modifiers, and more diverse dependency paths to prevent 
    empty sentence structures.
    """
    def generate_abstract(self, reduced_G):
        sentences = []
        visited_edges = set()
        
        for u, v, data in list(reduced_G.edges(data=True)):
            edge_key = (u, v, data.get('relation'))
            if edge_key in visited_edges:
                continue
            
            relation = data.get('relation')
            pos_u = reduced_G.nodes[u].get('pos') if u in reduced_G.nodes else None
            pos_v = reduced_G.nodes[v].get('pos') if v in reduced_G.nodes else None
            
            # Check for active subject, passive subject, or general modifier connections to a VERB
            if relation in ["nsubj", "nsubjpass", "dep"] and pos_v == "VERB":
                # u is subject/agent, v is verb
                # Search for direct objects, prepositional objects, or attributes of verb v
                obj_node = None
                for succ in reduced_G.successors(v):
                    for edge_idx in reduced_G[v][succ]:
                        edge_data = reduced_G[v][succ][edge_idx]
                        if edge_data.get('relation') in ["dobj", "pobj", "attr", "oprd"]:
                            obj_node = succ
                            break
                
                if obj_node:
                    if relation == "nsubjpass":
                        sentence = f"{u.capitalize()} was {v} by {obj_node}."
                    else:
                        sentence = f"{u.capitalize()} {v} {obj_node}."
                else:
                    if relation == "nsubjpass":
                        sentence = f"{u.capitalize()} was {v}."
                    else:
                        sentence = f"{u.capitalize()} {v}."
                
                if sentence not in sentences:
                    sentences.append(sentence)
            
            visited_edges.add(edge_key)
            
        if not sentences:
            # Enhanced template-based assertion fallback using highly ranked semantic concepts
            top_nodes = sorted(reduced_G.nodes(data=True), key=lambda x: x[1].get('score', 0), reverse=True)[:5]
            words = [node[0] for node in top_nodes]
            if words:
                if len(words) > 1:
                    sentences.append(f"Important themes relate to the context of {', '.join(words[:-1])} and {words[-1]}.")
                else:
                    sentences.append(f"Primary focus points center around {words[0]}.")
                
        return " ".join(sentences)


class DGNNGraphOptimizer:
    """
    Implements a Dynamic Graph Neural Network (DGNN) optimization pipeline
    for semantic graphs, featuring semantic weight initialization, Sampson 
    distance estimation, and iterative structural/feature updates.
    """
    def __init__(self, n_iter=5, sam_threshold=0.6):
        # Instantiate our newly constructed Robust Semantic Graph Reducer subclass
        self.reducer = RobustSemanticGraphReducer()
        self.nlp = self.reducer.nlp
        self.n_iter = n_iter
        self.sam_threshold = sam_threshold

    def _build_cooccurrence_features(self, G, doc):
        nodes = list(G.nodes())
        sentences = list(doc.sents)
        feature_matrix = {}
        dim = max(len(sentences), 10)

        for node in nodes:
            vector = np.zeros(dim)
            for i, sent in enumerate(sentences):
                if node in sent.text.lower() and i < dim:
                    vector[i] = 1.0
            feature_matrix[node] = vector

        return feature_matrix, dim

    def _estimate_semantic_transition_matrix(self, feature_matrix, dim):
        vectors = list(feature_matrix.values())
        if len(vectors) < 2:
            return np.eye(dim)
        
        X = np.array(vectors)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        rank = min(5, Vt.shape[0])
        M = Vt[:rank].T @ Vt[:rank]
        return M

    def _calculate_sampson_distance(self, vec_u, vec_v, M):
        numerator = (vec_u.T @ M @ vec_v) ** 2
        Mp1 = M @ vec_v
        MTp0 = M.T @ vec_u
        denominator = (Mp1[0]**2 + Mp1[1]**2 + MTp0[0]**2 + MTp0[1]**2) + 1e-6
        return numerator / denominator

    def optimize_graph(self, G, text):
        doc = self.nlp(text)
        optimized_G = G.copy()
        
        if len(optimized_G.nodes()) == 0:
            return optimized_G

        feature_matrix, dim = self._build_cooccurrence_features(optimized_G, doc)
        M = self._estimate_semantic_transition_matrix(feature_matrix, dim)

        # Normalize semantic scores
        scores = [data.get('score', 1.0) for _, data in optimized_G.nodes(data=True)]
        min_s, max_s = min(scores), max(scores)
        score_range = (max_s - min_s) if max_s != min_s else 1.0
        
        node_alphas = {}
        for node, data in optimized_G.nodes(data=True):
            raw_score = data.get('score', 1.0)
            node_alphas[node] = (raw_score - min_s) / score_range

        # Edge Initialization
        edge_weights = {}
        for u, v, key in optimized_G.edges(keys=True):
            vec_u, vec_v = feature_matrix[u], feature_matrix[v]
            norm_u, norm_v = np.linalg.norm(vec_u), np.linalg.norm(vec_v)
            similarity = (np.dot(vec_u, vec_v) / (norm_u * norm_v)) if (norm_u > 0 and norm_v > 0) else 0.0
            edge_weights[(u, v, key)] = max(float(similarity), 0.0)

        # DGNN Optimization Iterations
        for _ in range(self.n_iter):
            edge_g = {}
            for u, v, key in optimized_G.edges(keys=True):
                vec_u, vec_v = feature_matrix[u], feature_matrix[v]
                d_sampson = self._calculate_sampson_distance(vec_u, vec_v, M)
                edge_g[(u, v, key)] = np.exp(-d_sampson / 0.5)

            node_g = {}
            for node in optimized_G.nodes():
                connected_g = [edge_g[e] for e in optimized_G.edges(node, keys=True) if e in edge_g]
                node_g[node] = np.mean(connected_g) if connected_g else 0.0

            for u, v, key in list(optimized_G.edges(keys=True)):
                w = edge_weights[(u, v, key)]
                alpha_u, alpha_v = node_alphas[u], node_alphas[v]
                g_u, g_v = node_g[u], node_g[v]
                
                w_prime = w * np.sqrt(alpha_u * alpha_v)
                w_double_prime = w_prime * np.sqrt(g_u * g_v)
                
                vec_u, vec_v = feature_matrix[u], feature_matrix[v]
                norm_u, norm_v = np.linalg.norm(vec_u), np.linalg.norm(vec_v)
                similarity = (np.dot(vec_u, vec_v) / (norm_u * norm_v)) if (norm_u > 0 and norm_v > 0) else 0.0
                
                edge_weights[(u, v, key)] = w_double_prime * max(float(similarity), 0.0)

            # --- Dynamic Adaptive Pruning with Semantic Protection ---
            if len(optimized_G.nodes()) > 5:
                g_values = [val for val in node_g.values() if val > 0.0]
                if g_values:
                    percentile_threshold = np.percentile(g_values, 30)
                    final_threshold = min(self.sam_threshold, percentile_threshold)
                else:
                    final_threshold = 0.0
            else:
                final_threshold = 0.0

            # Pruning logic incorporating semantic threshold bounds
            for node in list(optimized_G.nodes()):
                alpha_u = node_alphas.get(node, 0.0)
                g_u = node_g.get(node, 0.0)
                
                # Protect nodes in the top tier of semantic significance (alpha >= 0.70)
                # from being pruned due to local structural sparsity.
                if g_u < final_threshold and alpha_u < 0.70:
                    optimized_G.remove_node(node)

        for u, v, key in optimized_G.edges(keys=True):
            optimized_G[u][v][key]['weight'] = edge_weights.get((u, v, key), 0.0)

        return optimized_G

    def summarize_optimized(self, text):
        if not text or not text.strip():
            return ""
        G = self.reducer.build_rich_semantic_graph(text)
        optimized_G = self.optimize_graph(G, text)
        reduced_G = self.reducer.reduce_graph(optimized_G)
        summary = self.reducer.generate_abstract(reduced_G)
        return summary


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    output_path = os.path.join(current_dir, "dgnn_optimized_results.md")

    try:
        data = load_parquet_data()
    except Exception as e:
        print(f"Failed to load dataset: {e}. Executing fallback assessment.")
        data = ["The quick brown fox jumps over the lazy dog. Dogs are loyal animals. Loyal animals make great companions."]

    if not isinstance(data, list):
        data = [data]

    baseline_reducer = SemanticGraphReducer()
    dgnn_optimizer = DGNNGraphOptimizer(n_iter=5, sam_threshold=0.6)

    print("Beginning comparative execution run...")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Dynamic Graph Semantic Optimization (DG-SGR) vs Baseline SGR\n\n")
        f.write("This report evaluates the performance of the DGNN optimization pipeline "
                "against the baseline Semantic Graph Reduction (SGR) model.\n\n")

        for index, record in enumerate(data[:3]):
            if isinstance(record, dict):
                text = record.get("text", "")
            else:
                text = str(record)

            if not text.strip():
                continue

            f.write(f"## Document {index + 1}\n\n")
            f.write("### Original Passage Excerpt\n")
            f.write(f"{text[:300]}{'...' if len(text) > 300 else ''}\n\n")

            baseline_summary = baseline_reducer.summarize(text)
            b_coherence = calculate_coherence(baseline_summary)
            b_wn_score = calculate_synonym_frequency_score(baseline_summary)

            optimized_summary = dgnn_optimizer.summarize_optimized(text)
            opt_coherence = calculate_coherence(optimized_summary)
            opt_wn_score = calculate_synonym_frequency_score(optimized_summary)

            f.write("### Summarization Artifacts\n\n")
            f.write(f"**Baseline SGR Summary:**\n> {baseline_summary if baseline_summary else '[No baseline summary generated]'}\n\n")
            f.write(f"**DGNN Optimized SGR Summary:**\n> {optimized_summary if optimized_summary else '[No optimized summary generated]'}\n\n")

            f.write("### Quantitative Evaluation\n\n")
            f.write("| Pipeline Variant | Sentence Coherence Similarity | Synonym Frequency Rank (WordNet) |\n")
            f.write("|---|---|---|\n")
            f.write(f"| Baseline SGR | {b_coherence:.4f} | {b_wn_score:.4f} |\n")
            f.write(f"| DGNN Optimized SGR | {opt_coherence:.4f} | {opt_wn_score:.4f} |\n\n")
            f.write("---\n\n")

    print(f"Comparative report successfully written to {output_path}")