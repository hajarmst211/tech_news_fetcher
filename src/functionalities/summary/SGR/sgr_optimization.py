import os
import sys
import itertools
import numpy as np


def expand_parameter_space(values, num_intermediates=3):
    """
    Takes a list of values and adds evenly spaced intermediate values
    between each consecutive pair. Returns the expanded sorted list.
    """
    values = sorted(values)
    expanded = [values[0]]
    for i in range(len(values) - 1):
        intermediates = np.linspace(values[i], values[i + 1], num=num_intermediates + 2)[1:-1]
        expanded.extend(intermediates)
    return sorted(set(expanded))

# Resolve path for data loader
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
sys.path.insert(0, os.path.join(current_dir, ".."))
sys.path.insert(0, current_dir)

# Handle potential import spelling variations dynamically
sgr_module = None
for module_name in ["semantic_graph_reduction"]:
    try:
        sgr_module = __import__(module_name)
        break
    except ImportError:
        continue

if sgr_module is None:
    raise ImportError(
        "Could not import the semantic graph reduction module. "
        "Ensure this script is placed in the same folder as your SGR python file."
    )

# Extract required classes, functions, and loader from the imported module
SemanticGraphReducer = sgr_module.SemanticGraphReducer
calculate_coherence = sgr_module.calculate_coherence
calculate_synonym_frequency_score = sgr_module.calculate_synonym_frequency_score
load_parquet_data = sgr_module.load_parquet_data


class ParameterizedSemanticGraphReducer(SemanticGraphReducer):
    """
    Subclass of SemanticGraphReducer that parameterizes the previously
    hardcoded values for hyperparameter optimization.
    """
    def __init__(self, similarity_threshold=0.7, top_n_fallback=5, prune_percentile=0.0):
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.top_n_fallback = top_n_fallback
        self.prune_percentile = prune_percentile

    def reduce_graph(self, G):
        """
        Overridden reduction phase utilizing parameter settings.
        """
        reduced_G = G.copy()
        
        # Optional structural pruning: remove lowest-scoring nodes before reduction
        if self.prune_percentile > 0.0:
            scores = [data.get('score', 1.0) for _, data in reduced_G.nodes(data=True)]
            if scores:
                threshold_score = np.percentile(scores, self.prune_percentile)
                nodes_to_remove = [node for node, data in reduced_G.nodes(data=True) if data.get('score', 1.0) < threshold_score]
                reduced_G.remove_nodes_from(nodes_to_remove)

        nodes = list(reduced_G.nodes(data=True))
        merged_nodes = set()

        for i in range(len(nodes)):
            node_u, data_u = nodes[i]
            if node_u in merged_nodes or not reduced_G.has_node(node_u):
                continue
                
            pos_u = data_u.get('pos')
            wn_pos_u = self._get_wordnet_pos(pos_u)
            synsets_u = sgr_module.wn.synsets(node_u, pos=wn_pos_u) if wn_pos_u else []
            
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
                synsets_v = sgr_module.wn.synsets(node_v, pos=wn_pos_v) if wn_pos_v else []
                
                if not synsets_v:
                    continue

                similarity = synsets_u[0].path_similarity(synsets_v[0])
                is_related = False
                
                hyper_u = set(synsets_u[0].closure(lambda s: s.hypernyms()))
                if synsets_v[0] in hyper_u:
                    is_related = True
                
                if pos_u == "VERB":
                    entailments_u = set(synsets_u[0].entailments())
                    if synsets_v[0] in entailments_u:
                        is_related = True
                
                # Apply parameterized similarity threshold here
                if (similarity and similarity > self.similarity_threshold) or is_related:
                    keep, discard = (node_u, node_v) if data_u['score'] >= data_v['score'] else (node_v, node_u)
                    
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
        Overridden generation phase utilizing parameterized fallback settings.
        """
        sentences = []
        visited_edges = set()
        
        for u, v, data in reduced_G.edges(data=True):
            edge_key = (u, v, data.get('relation'))
            if edge_key in visited_edges:
                continue
            
            relation = data.get('relation')
            pos_u = reduced_G.nodes[u].get('pos')
            pos_v = reduced_G.nodes[v].get('pos')
            
            if relation == "nsubj" and pos_v == "VERB":
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
            # Apply parameterized fallback node count here
            top_nodes = sorted(reduced_G.nodes(data=True), key=lambda x: x[1].get('score', 0), reverse=True)[:self.top_n_fallback]
            words = [node[0] for node in top_nodes]
            if words:
                sentences.append(f"Focus points include: {', '.join(words)}.")
            else:
                sentences.append("")
                
        return " ".join(sentences)


def evaluate_configuration(data_sample, similarity_threshold, top_n_fallback, prune_percentile):
    """
    Runs the reducer with specified parameters over the dataset sample 
    and returns average coherence and synonym frequency metrics.
    """
    reducer = ParameterizedSemanticGraphReducer(
        similarity_threshold=similarity_threshold,
        top_n_fallback=top_n_fallback,
        prune_percentile=prune_percentile
    )
    
    coherence_scores = []
    synonym_scores = []
    
    for record in data_sample:
        if isinstance(record, dict):
            text = record.get("text", "")
        elif isinstance(record, str):
            text = record
        else:
            text = str(record)
            
        if not text.strip():
            continue
            
        summary = reducer.summarize(text)
        
        coherence_scores.append(calculate_coherence(summary))
        synonym_scores.append(calculate_synonym_frequency_score(summary))
        
    avg_coherence = float(np.mean(coherence_scores)) if coherence_scores else 0.0
    avg_synonym = float(np.mean(synonym_scores)) if synonym_scores else 0.0
    
    return avg_coherence, avg_synonym


def run_grid_search():
    try:
        data = load_parquet_data()
    except Exception as e:
        print(f"Data loader error: {e}. Falling back to default baseline text.")
        data = ["The quick brown fox jumps over the lazy dog. Dogs are loyal animals. Loyal animals make great companions."]

    if not isinstance(data, list):
        data = [data]

    # Limit search optimization evaluation to the first 5 records to balance run-time and representation
    evaluation_sample = data[:5]

    # Define base hyperparameter spaces (expanded automatically)
    similarity_thresholds = expand_parameter_space([0.05, 0.9, 0.20])
    top_n_fallbacks = expand_parameter_space([1, 2, 5])
    prune_percentiles = expand_parameter_space([35.0, 44.0, 52.0])

    combinations = list(itertools.product(similarity_thresholds, top_n_fallbacks, prune_percentiles))
    
    results = []
    print(f"Starting optimization grid search ({len(combinations)} total configurations)...")
    
    for i, (sim, top_n, prune) in enumerate(combinations, start=1):
        print(f"Trial {i}/{len(combinations)}: Similarity={sim}, Fallback Limit={top_n}, Pruning Percentile={prune}%")
        avg_coherence, avg_synonym = evaluate_configuration(evaluation_sample, sim, top_n, prune)
        
        # Calculate a normalized composite score: Coherence carries 60% weight, Synonym score carries 40% (log scale to bound outliers)
        log_synonym = np.log1p(avg_synonym)
        composite_score = (avg_coherence * 0.6) + (min(log_synonym / 5.0, 1.0) * 0.4)
        
        results.append({
            "trial": i,
            "similarity_threshold": sim,
            "top_n_fallback": top_n,
            "prune_percentile": prune,
            "coherence": avg_coherence,
            "synonym_score": avg_synonym,
            "composite": composite_score
        })

    # Sort results by descending composite performance
    sorted_results = sorted(results, key=lambda x: x["composite"], reverse=True)
    best_config = sorted_results[0]

    # Write the formatted output report directly to sgr_optimization_output.md
    output_path = os.path.join(current_dir, "sgr_optimization_output.md")
    
    markdown_content = f"""# Hyperparameter Optimization Results

This document presents the systematic evaluation of structural configurations for the Semantic Graph Reduction (SGR) Summarizer.

---

## Optimal Configuration Found
*   **Best Score:** `{best_config['composite']:.4f}`
*   **Similarity Threshold:** `{best_config['similarity_threshold']}`
*   **Fallback Limit:** `{best_config['top_n_fallback']}`
*   **Prune Percentile:** `{best_config['prune_percentile']}%`
*   **Average Sentence Coherence Score:** `{best_config['coherence']:.4f}`
*   **Average Synonym Frequency Score:** `{best_config['synonym_score']:.4f}`
*   **Composite Benchmark Score:** `{best_config['composite']:.4f}`

---

## Complete Trial Results Table

| Trial | Similarity Threshold | Fallback Limit | Pruning % | Coherence Score | Synonym Freq Score | Composite Performance |
| :---: | :------------------: | :------------: | :-------: | :-------------: | :----------------: | :-------------------: |
"""

    for r in sorted_results:
        markdown_content += f"| {r['trial']} | {r['similarity_threshold']} | {r['top_n_fallback']} | {r['prune_percentile']}% | {r['coherence']:.4f} | {r['synonym_score']:.4f} | {r['composite']:.4f} |\n"

    markdown_content += """
---
*Note: The composite performance is calculated using a weighted balance prioritizing sentence structural coherence (60%) and logarithmic WordNet synonym frequency distribution (40%).*
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"\nOptimization complete. Output successfully saved to: {output_path}")


if __name__ == "__main__":
    run_grid_search()