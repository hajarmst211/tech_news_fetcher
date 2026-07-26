import os
import sys
import itertools
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import sent_tokenize

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from gasum_summarizer import compute_f1_score, load_parquet_data

class OptimizedGeneticAlgorithmSummarizer:
    def __init__(self, sentences, pop_size=50, generations=100, p_crossover=0.7, 
                 p_mutation=0.15, target_len=3, tournament_size=16):
        self.sentences = sentences
        self.num_sentences = len(sentences)
        self.pop_size = pop_size
        self.generations = generations
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.target_len = target_len
        self.tournament_size = min(tournament_size, pop_size - 1)

    def _initialize_population(self):
        population = []
        random_pop_size = int(self.pop_size * 0.7)
        for _ in range(random_pop_size):
            chromosome = np.zeros(self.num_sentences, dtype=int)
            k = min(self.target_len, self.num_sentences)
            indices = np.random.choice(self.num_sentences, k, replace=False)
            chromosome[indices] = 1
            population.append(chromosome)
            
        biased_pop_size = self.pop_size - random_pop_size
        lead_pool_limit = min(5, self.num_sentences)
        
        for _ in range(biased_pop_size):
            chromosome = np.zeros(self.num_sentences, dtype=int)
            k = min(self.target_len, self.num_sentences)
            indices = np.random.choice(lead_pool_limit, min(k, lead_pool_limit), replace=False)
            chromosome[indices] = 1
            if len(indices) < k:
                remaining_indices = np.setdiff1d(np.arange(self.num_sentences), indices)
                extra = np.random.choice(remaining_indices, k - len(indices), replace=False)
                chromosome[extra] = 1
            population.append(chromosome)
            
        return np.array(population)

    def _tournament_selection(self, population, fitnesses):
        selected_indices = np.random.choice(len(population), self.tournament_size, replace=True)
        best_idx = selected_indices[np.argmax(fitnesses[selected_indices])]
        return population[best_idx].copy()

    def _crossover(self, parent1, parent2):
        if self.num_sentences > 1 and np.random.rand() < self.p_crossover:
            point = np.random.randint(1, self.num_sentences)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2
        return parent1.copy(), parent2.copy()

    def _mutate(self, chromosome):
        if np.random.rand() < self.p_mutation:
            ones = np.where(chromosome == 1)[0]
            zeros = np.where(chromosome == 0)[0]
            if len(ones) > 0 and len(zeros) > 0:
                idx_to_deactivate = np.random.choice(ones)
                idx_to_activate = np.random.choice(zeros)
                chromosome[idx_to_deactivate] = 0
                chromosome[idx_to_activate] = 1
        return chromosome

    def run(self, fitness_fn):
        if self.num_sentences <= self.target_len:
            return np.ones(self.num_sentences, dtype=int)

        population = self._initialize_population()
        best_individual = None
        best_fitness = -float('inf')

        for gen in range(self.generations):
            fitnesses = np.array([fitness_fn(ind) for ind in population])

            max_idx = np.argmax(fitnesses)
            if fitnesses[max_idx] > best_fitness:
                best_fitness = fitnesses[max_idx]
                best_individual = population[max_idx].copy()

            next_population = []
            if best_individual is not None:
                next_population.append(best_individual.copy())
            
            while len(next_population) < self.pop_size:
                p1 = self._tournament_selection(population, fitnesses)
                p2 = self._tournament_selection(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                next_population.append(self._mutate(c1))
                if len(next_population) < self.pop_size:
                    next_population.append(self._mutate(c2))

            population = np.array(next_population)

        return best_individual


class OptimizedGaSUMPipeline:
    def __init__(self, doc_text, model, tokenizer, device, target_len=3):
        self.doc_text = doc_text
        self.sentences = sent_tokenize(doc_text)
        self.target_len = target_len
        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        
        self.sentence_reprs = self._precompute_sentence_representations()
        self.doc_repr = self._get_bert_representation(self.doc_text).cpu()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _get_bert_representation(self, text):
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512, 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            pooled_representation = self._mean_pooling(outputs, inputs['attention_mask'])
        return pooled_representation

    def _precompute_sentence_representations(self):
        reprs = []
        for sentence in self.sentences:
            sentence_vector = self._get_bert_representation(sentence).cpu()
            reprs.append(sentence_vector.squeeze(0))
        return torch.stack(reprs) if reprs else torch.empty(0)

    def _decode_summary(self, chromosome):
        selected_sentences = [self.sentences[i] for i, gene in enumerate(chromosome) if gene == 1]
        return " ".join(selected_sentences)


class GridSearchPipeline(OptimizedGaSUMPipeline):
    def fitness_gasum_tuned(self, chromosome, alpha=1.0, beta=0.2):
        selected_indices = np.where(chromosome == 1)[0]
        n_selected = len(selected_indices)
        if n_selected == 0:
            return -100.0

        selected_reprs = self.sentence_reprs[selected_indices]
        cand_repr = torch.mean(selected_reprs, dim=0, keepdim=True)
        coverage = torch.cosine_similarity(self.doc_repr, cand_repr, dim=1).item()
        
        redundancy = 0.0
        if n_selected > 1:
            norms = torch.norm(selected_reprs, p=2, dim=1, keepdim=True)
            normalized_reprs = selected_reprs / torch.clamp(norms, min=1e-9)
            sim_matrix = torch.mm(normalized_reprs, normalized_reprs.t())
            
            triu_indices = torch.triu_indices(n_selected, n_selected, offset=1)
            pairwise_similarities = sim_matrix[triu_indices[0], triu_indices[1]]
            redundancy = pairwise_similarities.mean().item()
            
        length_penalty = abs(n_selected - self.target_len) * 0.15
        return (alpha * coverage) - (beta * redundancy) - length_penalty

    def generate_summary_tuned(self, pop_size, generations, p_crossover, p_mutation, tournament_size, beta):
        if len(self.sentences) == 0:
            return ""
        
        ga = OptimizedGeneticAlgorithmSummarizer(
            self.sentences, 
            pop_size=pop_size, 
            generations=generations, 
            target_len=self.target_len,
            p_crossover=p_crossover,
            p_mutation=p_mutation,
            tournament_size=tournament_size
        )
        
        fitness_fn = lambda chrom: self.fitness_gasum_tuned(chrom, alpha=1.0, beta=beta)
        best_chrom = ga.run(fitness_fn)
        return self._decode_summary(best_chrom)


def run_grid_search(validation_dataset, param_grid, model, tokenizer, device, target_length=3):
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Total configurations to evaluate: {len(experiments)}")
    print(f"Evaluating on {len(validation_dataset)} validation documents...\n")
    
    best_score = -1.0
    best_config = None
    results = []

    for config_idx, config in enumerate(experiments):
        scores = []
        
        for doc_idx, item in enumerate(validation_dataset):
            doc_text = item.get('document', '')
            ref_text = item.get('abstract', '')
            
            sentences = sent_tokenize(doc_text)
            if len(sentences) <= target_length:
                continue
                
            pipeline = GridSearchPipeline(doc_text, model, tokenizer, device, target_len=target_length)
            
            summary = pipeline.generate_summary_tuned(
                pop_size=config['pop_size'],
                generations=config['generations'],
                p_crossover=config['p_crossover'],
                p_mutation=config['p_mutation'],
                tournament_size=config['tournament_size'],
                beta=config['beta']
            )
            
            f1 = compute_f1_score(summary, ref_text)
            scores.append(f1)
            
        mean_score = np.mean(scores) if scores else 0.0
        results.append({**config, "mean_f1": mean_score})
        
        if mean_score > best_score:
            best_score = mean_score
            best_config = config
            
        print(f"Config {config_idx + 1}/{len(experiments)}: {config}")
        print(f"Current F1: {mean_score:.4f} | Best F1 So Far: {best_score:.4f}\n")

    df_results = pd.DataFrame(results)

    output_path = os.path.join(current_dir, "gasum_optimization_results.md")

    with open(output_path, "w") as f:
        f.write("# GaSUM Hyperparameter Optimization Results\n\n")
        f.write("## Best Configuration Found\n\n")
        for k, v in best_config.items():
            f.write(f"- **{k}**: {v}\n")
        f.write(f"- **Best Mean F1**: {best_score:.4f}\n\n")
        f.write("## All Run Configurations\n\n")
        
        headers = list(df_results.columns)
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for idx, row in df_results.iterrows():
            formatted_values = []
            for h in headers:
                val = row[h]
                if isinstance(val, float):
                    formatted_values.append(f"{val:.4f}")
                else:
                    formatted_values.append(str(val))
            f.write("| " + " | ".join(formatted_values) + " |\n")
        
        f.write("\n## Summary\n\n")
        f.write(f"**Best Mean F1 Score: {best_score:.4f}**\n\n")
        f.write("**Best Parameter Values:**\n\n")
        f.write("| Parameter | Value |\n")
        f.write("| --- | --- |\n")
        for k, v in best_config.items():
            f.write(f"| {k} | {v} |\n")

    print("\n" + "="*50)
    print("GRID SEARCH COMPLETE")
    print("="*50)
    print(f"Best Mean F1: {best_score:.4f}")
    print(f"Best Hyperparameters: {best_config}")
    print("="*50)
    
    return df_results, best_config


if __name__ == "__main__":
    df = load_parquet_data()
    num_docs_for_val = 3  
    val_data = df.head(num_docs_for_val).to_dict(orient="records")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    model.eval()

    search_space = {
        "pop_size": [20, 40],
        "generations": [10, 15],
        "p_crossover": [0.6, 0.8],
        "p_mutation": [0.1, 0.2],
        "tournament_size": [8, 16],
        "beta": [0.1, 0.3]
    }
    
    df_results, best_hyperparams = run_grid_search(val_data, search_space, model, tokenizer, device, target_length=3)