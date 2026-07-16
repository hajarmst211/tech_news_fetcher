import os
import random
import numpy as np
import pandas as pd
import optuna
import concurrent.futures
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from ga_summarizer import SummarizationPipeline, compute_f1_score

def set_seed(seed):
    """Utility to set random states for stochastic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

class CustomGeneticAlgorithm:
    def __init__(self, num_sentences, pop_size=20, generations=20, p_crossover=0.8, p_mutation=0.05, elitism_ratio=0.1, tournament_size=3):
        self.num_sentences = num_sentences
        self.pop_size = pop_size
        self.generations = generations
        self.p_crossover = p_crossover
        self.initial_p_mutation = p_mutation
        self.elitism_ratio = elitism_ratio
        self.tournament_size = tournament_size

    def run(self, fitness_fn):
        # 2. Fitness Caching (Memoization)
        fitness_cache = {}
        def get_fitness(chrom):
            chrom_tuple = tuple(chrom)
            if chrom_tuple not in fitness_cache:
                fitness_cache[chrom_tuple] = fitness_fn(chrom)
            return fitness_cache[chrom_tuple]

        # Initialize Population
        population = []
        for _ in range(self.pop_size):
            chrom = np.zeros(self.num_sentences, dtype=int)
            if self.num_sentences > 0:
                initial_indices = np.random.choice(
                    self.num_sentences, 
                    min(5, self.num_sentences), 
                    replace=False
                )
                chrom[initial_indices] = 1
            population.append(chrom)
        population = np.array(population)

        best_chrom = population[0].copy()
        best_fit = -float('inf')

        for gen in range(self.generations):
            # 3. Adaptive Mutation: Decay mutation rate linearly over generations
            current_p_mutation = self.initial_p_mutation * (1.0 - (gen / self.generations))
            current_p_mutation = max(current_p_mutation, 0.01)

            # Evaluate fitnesses
            fitnesses = np.array([get_fitness(ind) for ind in population])

            # Update best tracker
            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > best_fit:
                best_fit = fitnesses[gen_best_idx]
                best_chrom = population[gen_best_idx].copy()

            # 1. Elitism: Preserve the best individuals
            num_elites = max(1, int(self.pop_size * self.elitism_ratio))
            elite_indices = np.argsort(fitnesses)[-num_elites:]
            elites = population[elite_indices].copy()

            next_pop = []
            for _ in range(self.pop_size - num_elites):
                # 3. Tournament Selection
                t_indices_1 = np.random.choice(self.pop_size, self.tournament_size, replace=False)
                parent1 = population[t_indices_1[np.argmax(fitnesses[t_indices_1])]]
                
                t_indices_2 = np.random.choice(self.pop_size, self.tournament_size, replace=False)
                parent2 = population[t_indices_2[np.argmax(fitnesses[t_indices_2])]]

                # Two-point crossover
                if random.random() < self.p_crossover and self.num_sentences > 2:
                    pt1 = random.randint(1, self.num_sentences - 2)
                    pt2 = random.randint(pt1 + 1, self.num_sentences - 1)
                    child = np.concatenate([parent1[:pt1], parent2[pt1:pt2], parent1[pt2:]])
                else:
                    child = parent1.copy()

                # Mutation
                for gene_idx in range(self.num_sentences):
                    if random.random() < current_p_mutation:
                        child[gene_idx] = 1 - child[gene_idx]

                next_pop.append(child)

            population = np.vstack([elites, np.array(next_pop)])

        # 4. Local Search (Hill Climbing / Memetic step on the final best chromosome)
        local_best_chrom = best_chrom.copy()
        local_best_fit = best_fit
        for idx in range(self.num_sentences):
            candidate = local_best_chrom.copy()
            candidate[idx] = 1 - candidate[idx]
            candidate_fit = get_fitness(candidate)
            if candidate_fit > local_best_fit:
                local_best_fit = candidate_fit
                local_best_chrom = candidate

        return local_best_chrom


class OptimizedSummarizationPipeline(SummarizationPipeline):
    def __init__(self, text, target_len=5, **kwargs):
        super().__init__(text, target_len=target_len)
        self.w_coverage_std = kwargs.get("w_coverage_std", 0.7)
        self.w_redundancy_std = kwargs.get("w_redundancy_std", 0.3)
        self.w_coverage_mcba = kwargs.get("w_coverage_mcba", 0.4)
        self.w_position_mcba = kwargs.get("w_position_mcba", 0.4)
        self.w_redundancy_mcba = kwargs.get("w_redundancy_mcba", 0.2)
        self.w_coverage_rpm = kwargs.get("w_coverage_rpm", 0.8)
        self.w_redundancy_rpm = kwargs.get("w_redundancy_rpm", 0.2)
        self.penalty_weight = kwargs.get("penalty_weight", 2.0)
        self.pop_size = kwargs.get("pop_size", 20)
        self.generations = kwargs.get("generations", 20)
        self.p_crossover = kwargs.get("p_crossover", 0.8)
        self.p_mutation = kwargs.get("p_mutation", 0.05)

        # 2. Precomputation of similarities to avoid runtime matrix operations
        if self.tfidf_matrix is not None and self.doc_vector is not None:
            self.sentence_doc_similarities = cosine_similarity(self.tfidf_matrix, self.doc_vector).flatten()
            self.pairwise_similarities = cosine_similarity(self.tfidf_matrix)
        else:
            self.sentence_doc_similarities = None
            self.pairwise_similarities = None

    def fitness_standard_ga(self, chromosome):
        selected_indices = np.where(chromosome == 1)[0]
        if len(selected_indices) == 0:
            return -100.0
        length_penalty = abs(len(selected_indices) - self.target_len) * self.penalty_weight
        if self.sentence_doc_similarities is None:
            return -length_penalty
        
        coverage = np.mean(self.sentence_doc_similarities[selected_indices])
        if len(selected_indices) > 1:
            pairwise_sub = self.pairwise_similarities[np.ix_(selected_indices, selected_indices)]
            tri_indices = np.triu_indices_from(pairwise_sub, k=1)
            redundancy = np.mean(pairwise_sub[tri_indices]) if len(tri_indices[0]) > 0 else 0
        else:
            redundancy = 0
        
        fitness = (self.w_coverage_std * coverage) - (self.w_redundancy_std * redundancy) - length_penalty
        return fitness

    def fitness_mcba_ga(self, chromosome):
        selected_indices = np.where(chromosome == 1)[0]
        if len(selected_indices) == 0:
            return -100.0
        length_penalty = abs(len(selected_indices) - self.target_len) * self.penalty_weight
        n = len(self.sentences)
        position_scores = []
        for idx in selected_indices:
            if idx == 0:
                pos_score = 1.0
            elif idx == n - 1:
                pos_score = 0.5
            else:
                pos_score = 1.0 / (idx + 1)
            position_scores.append(pos_score)
        avg_position_score = np.mean(position_scores)
        
        if self.sentence_doc_similarities is None:
            return (self.w_position_mcba * avg_position_score) - length_penalty
        
        coverage = np.mean(self.sentence_doc_similarities[selected_indices])
        if len(selected_indices) > 1:
            pairwise_sub = self.pairwise_similarities[np.ix_(selected_indices, selected_indices)]
            tri_indices = np.triu_indices_from(pairwise_sub, k=1)
            redundancy = np.mean(pairwise_sub[tri_indices]) if len(tri_indices[0]) > 0 else 0
        else:
            redundancy = 0
        
        fitness = (self.w_coverage_mcba * coverage) + (self.w_position_mcba * avg_position_score) - (self.w_redundancy_mcba * redundancy) - length_penalty
        return fitness

    def fitness_rpm_ga(self, chromosome, pattern_matrix, total_patterns):
        selected_indices = np.where(chromosome == 1)[0]
        if len(selected_indices) == 0:
            return -100.0
        length_penalty = abs(len(selected_indices) - self.target_len) * self.penalty_weight
        if total_patterns == 0:
            return self.fitness_standard_ga(chromosome)
        
        selected_patterns = pattern_matrix[selected_indices]
        covered_patterns = np.any(selected_patterns, axis=0).sum()
        pattern_coverage_ratio = covered_patterns / total_patterns
        if len(selected_indices) > 1:
            total_elements = len(selected_indices) * total_patterns
            redundancy = (selected_patterns.sum() - covered_patterns) / max(total_elements, 1)
        else:
            redundancy = 0
            
        fitness = (self.w_coverage_rpm * pattern_coverage_ratio) - (self.w_redundancy_rpm * redundancy) - length_penalty
        return fitness

    def generate_summary(self, method="standard", **kwargs):
        if len(self.sentences) == 0:
            return ""
        
        ga = CustomGeneticAlgorithm(
            num_sentences=len(self.sentences),
            pop_size=self.pop_size,
            generations=self.generations,
            p_crossover=self.p_crossover,
            p_mutation=self.p_mutation,
            elitism_ratio=0.1,
            tournament_size=3
        )
        
        if method == "standard":
            best_chrom = ga.run(self.fitness_standard_ga)
        elif method == "mcba":
            best_chrom = ga.run(self.fitness_mcba_ga)
        elif method == "rpm":
            pattern_matrix, total_patterns = self._extract_repetitive_patterns()
            best_chrom = ga.run(lambda chrom: self.fitness_rpm_ga(chrom, pattern_matrix, total_patterns))
        else:
            raise ValueError("Unknown method")
            
        return self._decode_summary(best_chrom)


def run_single_evaluation_task(args):
    """
    Self-contained worker task running in a distinct process.
    Handles the evaluation of one document/seed combination.
    """
    doc_idx, doc_text, ref_text, seed, params = args
    set_seed(seed)
    sentences = sent_tokenize(doc_text)
    if len(sentences) <= 5:
        return doc_idx, seed, 0.0, 0.0, 0.0

    pipeline = OptimizedSummarizationPipeline(
        doc_text,
        target_len=5,
        pop_size=params["pop_size"],
        generations=20,
        p_crossover=params["p_crossover"],
        p_mutation=params["p_mutation"],
        w_coverage_std=params["w_coverage"],
        w_redundancy_std=0.3,
        w_coverage_mcba=params["w_coverage"],
        w_position_mcba=params["w_position_mcba"],
        w_redundancy_mcba=0.2,
        w_coverage_rpm=params["w_coverage"],
        w_redundancy_rpm=0.2,
        penalty_weight=params["penalty_weight"]
    )

    s_std = pipeline.generate_summary(method="standard")
    f1_std = compute_f1_score(s_std, ref_text)

    s_mcba = pipeline.generate_summary(method="mcba")
    f1_mcba = compute_f1_score(s_mcba, ref_text)

    s_rpm = pipeline.generate_summary(method="rpm")
    f1_rpm = compute_f1_score(s_rpm, ref_text)

    return doc_idx, seed, f1_std, f1_mcba, f1_rpm


def objective(trial):
    params = {
        "pop_size": trial.suggest_categorical("pop_size", [15, 20, 25]),
        "w_coverage": trial.suggest_float("w_coverage", 0.45, 0.55, step=0.05),
        "penalty_weight": trial.suggest_float("penalty_weight", 0.10, 1.10, step=0.50),
        "p_crossover": trial.suggest_float("p_crossover", 0.60, 0.80, step=0.10),
        "p_mutation": trial.suggest_float("p_mutation", 0.07, 0.11, step=0.02),
        "w_position_mcba": trial.suggest_float("w_position_mcba", 0.50, 0.80, step=0.15),
    }

    seeds = [42, 100, 2023]
    
    # Pack parameters and tasks for multiprocessing
    tasks = []
    for i, item in enumerate(ds_val):
        for seed in seeds:
            tasks.append((i, item["document"], item["abstract"], seed, params))

    f1_stds = []
    f1_mcbas = []
    f1_rpms = []
    
    print(f"\n--- Starting Trial {trial.number} ---")
    
    # 4. Parallelization: Run document evaluations across multiple cores
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(run_single_evaluation_task, tasks))

    for doc_idx, seed, f1_std, f1_mcba, f1_rpm in results:
        if f1_std == 0.0 and f1_mcba == 0.0 and f1_rpm == 0.0:
            continue
        f1_stds.append(f1_std)
        f1_mcbas.append(f1_mcba)
        f1_rpms.append(f1_rpm)

    mean_std = np.mean(f1_stds) if f1_stds else 0.0
    mean_mcba = np.mean(f1_mcbas) if f1_mcbas else 0.0
    mean_rpm = np.mean(f1_rpms) if f1_rpms else 0.0
    overall_mean = (mean_std + mean_mcba + mean_rpm) / 3.0

    with open(md_file, "a") as f:
        f.write(
            f"| {trial.number} | {params['pop_size']} | {params['p_crossover']:.3f} | {params['p_mutation']:.3f} | "
            f"{params['w_coverage']:.2f} | {params['w_position_mcba']:.2f} | {params['penalty_weight']:.2f} | "
            f"{mean_std:.4f} | {mean_mcba:.4f} | {mean_rpm:.4f} | {overall_mean:.4f} |\n"
        )

    return overall_mean


if __name__ == "__main__":
    import urllib.request

    local_path = "train-00000-of-00015.parquet"
    if not os.path.exists(local_path):
        url = "https://huggingface.co/datasets/ccdv/arxiv-summarization/resolve/main/document/train-00000-of-00015.parquet"
        print("Local file not found. Starting dataset download...")
        urllib.request.urlretrieve(url, local_path)
        print("\nDownload finished.")

    print("Loading data from local storage...")
    df = pd.read_parquet(local_path, columns=["article", "abstract"])
    df = df.rename(columns={"article": "document"})
    
    global ds_val, md_file
    ds_val = df.iloc[1000:1005].to_dict(orient="records")
    md_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimisation_output.md")

    with open(md_file, "w") as f:
        f.write(
            "| Trial | Pop Size | P Crossover | P Mutation | W Coverage | W Pos MCBA | Penalty Weight | Mean Std (Multi-Seed) | Mean MCBA (Multi-Seed) | Mean RPM (Multi-Seed) | Overall Mean |\n"
        )
        f.write(
            "|---|---|---|---|---|---|---|---|---|---|---|\n"
        )

    search_space = {
        "pop_size": [15, 20, 25],
        "w_coverage": [0.45, 0.50, 0.55],
        "penalty_weight": [0.1, 0.6, 1.1],
        "p_crossover": [0.60, 0.70, 0.80],
        "p_mutation": [0.07, 0.09, 0.11],
        "w_position_mcba": [0.50, 0.65, 0.80],
    }

    total_combinations = np.prod([len(v) for v in search_space.values()])
    print(f"\nInitialized Narrow Grid Search Space with {total_combinations} total combinations.")
    print("Each combination will be evaluated across 5 documents and 3 random seeds.")
    print("Press Ctrl+C to stop the process early. Outcomes are appended progressively to optimisation_output.md.\n")

    study = optuna.create_study(
        sampler=optuna.samplers.GridSampler(search_space), 
        direction="maximize"
    )
    
    study.optimize(objective, n_trials=500)

    print("\n" + "="*50)
    print("NARROW GRID SEARCH COMPLETED")
    print("="*50)
    print("Best parameters found:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    print(f"Best multi-seed overall mean F1 score: {study.best_value:.4f}")