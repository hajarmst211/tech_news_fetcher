import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import collections
import pandas as pd

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True, download_dir='/home/hajora/nltk_data')
nltk.download('stopwords', quiet=True)



class GeneticAlgorithmSummarizer:
    def __init__(self, sentences, pop_size=50, generations=100, p_crossover=0.8, p_mutation=0.15, target_len=3):
        self.sentences = sentences
        self.num_sentences = len(sentences)
        self.pop_size = pop_size
        self.generations = generations
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.target_len = target_len

    def _initialize_population(self):
        """Generates random binary chromosomes with roughly target_len ones."""
        population = []
        for _ in range(self.pop_size):
            chromosome = np.zeros(self.num_sentences, dtype=int)
            indices = np.random.choice(self.num_sentences, self.target_len, replace=False)
            chromosome[indices] = 1
            population.append(chromosome)
        return np.array(population)

    def _tournament_selection(self, population, fitnesses, k=3):
        """Selects the best individual from a random sample of size k."""
        selected_indices = np.random.choice(len(population), k, replace=False)
        best_idx = selected_indices[np.argmax(fitnesses[selected_indices])]
        return population[best_idx].copy()

    def _crossover(self, parent1, parent2):
        """Applies single-point crossover."""
        if np.random.rand() < self.p_crossover:
            point = np.random.randint(1, self.num_sentences)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2
        return parent1.copy(), parent2.copy()

    def _mutate(self, chromosome):
        """Applies bit-flip mutation."""
        for i in range(self.num_sentences):
            if np.random.rand() < self.p_mutation:
                chromosome[i] = 1 - chromosome[i]
        return chromosome

    def run(self, fitness_fn):
        """Runs the genetic algorithm using a custom fitness function."""
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


class SummarizationPipeline:
    def __init__(self, text, target_len=5):
        self.sentences = sent_tokenize(text)
        self.target_len = target_len
        self.stop_words = set(stopwords.words('english'))
        
        self.vectorizer = TfidfVectorizer(stop_words='english')
        if len(self.sentences) > 1:
            try:
                self.tfidf_matrix = self.vectorizer.fit_transform(self.sentences)
                self.doc_vector = np.asarray(self.tfidf_matrix.mean(axis=0))
            except ValueError:
                self.tfidf_matrix = None
                self.doc_vector = None
        else:
            self.tfidf_matrix = None
            self.doc_vector = None

    def _decode_summary(self, chromosome):
        """Converts a binary chromosome back into a text summary string."""
        selected_sentences = [self.sentences[i] for i, gene in enumerate(chromosome) if gene == 1]
        return " ".join(selected_sentences)

    # STANDARD EXTRACTIVE GA 
    def fitness_standard_ga(self, chromosome):
        """Balances semantic coverage (TF-IDF) and redundancy mitigation."""
        selected_indices = np.where(chromosome == 1)[0]
        if len(selected_indices) == 0:
            return -100.0  # Heavy penalty for empty selection

        length_penalty = abs(len(selected_indices) - self.target_len) * 2.0

        if self.tfidf_matrix is None or self.doc_vector is None:
            return -length_penalty

        # Average similarity of selected sentences to entire document
        selected_vectors = self.tfidf_matrix[selected_indices]
        coverage = np.mean(cosine_similarity(selected_vectors, self.doc_vector))

        # Average pairwise similarity among selected sentences
        if len(selected_indices) > 1:
            pairwise_sim = cosine_similarity(selected_vectors)
            # Take the upper triangle excluding diagonal to get unique pairs
            tri_indices = np.triu_indices_from(pairwise_sim, k=1)
            redundancy = np.mean(pairwise_sim[tri_indices]) if len(tri_indices[0]) > 0 else 0
        else:
            redundancy = 0

        fitness = (0.7 * coverage) - (0.3 * redundancy) - length_penalty
        return fitness

    # MCBA + GA 
    def fitness_mcba_ga(self, chromosome):
        """Incorporates structural sentence position rankings into the fitness evaluation."""
        selected_indices = np.where(chromosome == 1)[0]
        if len(selected_indices) == 0:
            return -100.0

        length_penalty = abs(len(selected_indices) - self.target_len) * 2.0

        # MCBA Position Weighting:
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

        if self.tfidf_matrix is None or self.doc_vector is None:
            return (0.4 * avg_position_score) - length_penalty

        # Standard Coverage and Redundancy
        selected_vectors = self.tfidf_matrix[selected_indices]
        coverage = np.mean(cosine_similarity(selected_vectors, self.doc_vector))

        if len(selected_indices) > 1:
            pairwise_sim = cosine_similarity(selected_vectors)
            tri_indices = np.triu_indices_from(pairwise_sim, k=1)
            redundancy = np.mean(pairwise_sim[tri_indices]) if len(tri_indices[0]) > 0 else 0
        else:
            redundancy = 0

        # combining positional adn semantic scoring
        fitness = (0.4 * coverage) + (0.4 * avg_position_score) - (0.2 * redundancy) - length_penalty
        return fitness

    # REPETITIVE PATTERN MINING (RPM) + GA 
    def _extract_repetitive_patterns(self, min_support=2):
        """Identifies words that appear in at least 'min_support' sentences."""
        tokenized_sentences = []
        for sent in self.sentences:
            tokens = [w.lower() for w in word_tokenize(sent) if w.isalnum() and w.lower() not in self.stop_words]
            tokenized_sentences.append(set(tokens))

        word_counts = collections.Counter()
        for sent_set in tokenized_sentences:
            for word in sent_set:
                word_counts[word] += 1


        frequent_patterns = [word for word, count in word_counts.items() if count >= min_support]
        
        matrix = np.zeros((len(self.sentences), len(frequent_patterns)))
        for i, sent_set in enumerate(tokenized_sentences):
            for j, pattern in enumerate(frequent_patterns):
                if pattern in sent_set:
                    matrix[i, j] = 1 
        return matrix, len(frequent_patterns)

    def fitness_rpm_ga(self, chromosome, pattern_matrix, total_patterns):
        """Optimizes coverage of identified frequent patterns with minimal overlap."""
        selected_indices = np.where(chromosome == 1)[0]
        if len(selected_indices) == 0:
            return -100.0

        length_penalty = abs(len(selected_indices) - self.target_len) * 2.0

        if total_patterns == 0:
            return self.fitness_standard_ga(chromosome)

        #  how many unique frequent patterns covered
        selected_patterns = pattern_matrix[selected_indices]
        covered_patterns = np.any(selected_patterns, axis=0).sum()
        pattern_coverage_ratio = covered_patterns / total_patterns

        if len(selected_indices) > 1:
            total_elements = len(selected_indices) * total_patterns
            redundancy = (selected_patterns.sum() - covered_patterns) / max(total_elements, 1)
        else:
            redundancy = 0

        fitness = (0.8 * pattern_coverage_ratio) - (0.2 * redundancy) - length_penalty
        return fitness

    def generate_summary(self, method='standard', pop_size=40, generations=50):
        if len(self.sentences) == 0:
            return ""
        ga = GeneticAlgorithmSummarizer(self.sentences, pop_size=pop_size, generations=generations, target_len=self.target_len)
        
        if method == 'standard':
            best_chrom = ga.run(self.fitness_standard_ga)
        elif method == 'mcba':
            best_chrom = ga.run(self.fitness_mcba_ga)
        elif method == 'rpm':
            pattern_matrix, total_patterns = self._extract_repetitive_patterns()
            best_chrom = ga.run(lambda chrom: self.fitness_rpm_ga(chrom, pattern_matrix, total_patterns))
        else:
            raise ValueError("Unknown method")
            
        return self._decode_summary(best_chrom)


# Token-Overlap F1

def compute_f1_score(candidate_summary, reference_summary):
    """Computes a token-overlap F1 score as an approximation of ROUGE-1."""
    stop_words = set(stopwords.words('english'))
    
    cand_tokens = [w.lower() for w in word_tokenize(candidate_summary) if w.isalnum() and w.lower() not in stop_words]
    ref_tokens = [w.lower() for w in word_tokenize(reference_summary) if w.isalnum() and w.lower() not in stop_words]

    if not cand_tokens or not ref_tokens:
        return 0.0

    cand_counter = collections.Counter(cand_tokens)
    ref_counter = collections.Counter(ref_tokens)

    overlap = sum((cand_counter & ref_counter).values())
    
    precision = overlap / len(cand_tokens)
    recall = overlap / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


if __name__ == "__main__":
    import os
    print("importing dataset")
    local_path = "train-00000-of-00015.parquet"
    if not os.path.exists(local_path):
        url = "https://huggingface.co/datasets/ccdv/arxiv-summarization/resolve/main/document/train-00000-of-00015.parquet"
        df = pd.read_parquet(url, columns=["document", "abstract"])
        df.to_parquet(local_path)
    else:
        df = pd.read_parquet(local_path)
    
    print("df found")

    ds = df.head(100).to_dict(orient="records")

    print("=== STARTING SUMMARY EVALUATION ===")
    # Configurable limits
    num_to_evaluate = 5  
    target_length = 5    
    
    scores_std = []
    scores_mcba = []
    scores_rpm = []

    print(f"=== Document Processing Initialized ({num_to_evaluate} documents) ===")

    processed_count = 0
    for idx, item in enumerate(ds):
        if processed_count >= num_to_evaluate:
            break

        doc_text = item['document']
        ref_text = item['abstract']

        sentences = sent_tokenize(doc_text)
        if len(sentences) <= target_length:
            continue

        print(f"\nProcessing Document {idx + 1}/{num_to_evaluate} (Length: {len(sentences)} sentences)")
        pipeline = SummarizationPipeline(doc_text, target_len=target_length)

        # Standard GA
        summary_std = pipeline.generate_summary(method='standard', pop_size=40, generations=30)
        f1_std = compute_f1_score(summary_std, ref_text)
        scores_std.append(f1_std)

        #  MCBA + GA
        summary_mcba = pipeline.generate_summary(method='mcba', pop_size=40, generations=30)
        f1_mcba = compute_f1_score(summary_mcba, ref_text)
        scores_mcba.append(f1_mcba)

        #  RPM + GA
        summary_rpm = pipeline.generate_summary(method='rpm', pop_size=40, generations=30)
        f1_rpm = compute_f1_score(summary_rpm, ref_text)
        scores_rpm.append(f1_rpm)

        print(f"  F1 Standard: {f1_std:.4f} | F1 MCBA: {f1_mcba:.4f} | F1 RPM: {f1_rpm:.4f}")
        processed_count += 1

    if scores_std:
        print("\n" + "="*50)
        print(f"COMPARATIVE EVALUATION OVER {processed_count} DATASET SAMPLES")
        print("="*50)
        print(f"Average F-Measure (Standard GA):               {np.mean(scores_std):.4f}")
        print(f"Average F-Measure (MCBA + GA):                 {np.mean(scores_mcba):.4f}")
        print(f"Average F-Measure (RPM + GA):                  {np.mean(scores_rpm):.4f}")
        print("="*50)
    else:
        print("\nNo valid documents were processed.")