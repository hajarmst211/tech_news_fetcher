import os
import sys
import collections
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Ensure NLTK resources are available
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Attempt to handle __file__ when running in environments like notebooks
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(current_dir, ".."))
except NameError:
    # Fallback if __file__ is not defined
    sys.path.insert(0, "..")

# Try to import, fallback if data_loader is not in the path
try:
    from data_loader import load_parquet_data
except ImportError:
    # Dummy placeholder function if data_loader is missing in the local environment
    def load_parquet_data():
        return pd.DataFrame([
            {"document": "This is the first sentence. This is the second sentence. This is the third sentence. This is the fourth sentence.", 
             "abstract": "This is a brief abstract of the document."}
        ] * 5)


class GeneticAlgorithmSummarizer:
    def __init__(self, sentences, pop_size=50, generations=100, p_crossover=0.6, 
                 p_mutation=0.2, target_len=3, tournament_size=32):
        self.sentences = sentences
        self.num_sentences = len(sentences)
        self.pop_size = pop_size
        self.generations = generations
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.target_len = target_len
        self.tournament_size = min(tournament_size, pop_size - 1)

    def _initialize_population(self):
        """Generates random binary chromosomes with roughly target_len ones."""
        population = []
        for _ in range(self.pop_size):
            chromosome = np.zeros(self.num_sentences, dtype=int)
            k = min(self.target_len, self.num_sentences)
            indices = np.random.choice(self.num_sentences, k, replace=False)
            chromosome[indices] = 1
            population.append(chromosome)
        return np.array(population)

    def _tournament_selection(self, population, fitnesses):
        """Selects the best individual from a random sample."""
        selected_indices = np.random.choice(len(population), self.tournament_size, replace=True)
        best_idx = selected_indices[np.argmax(fitnesses[selected_indices])]
        return population[best_idx].copy()

    def _crossover(self, parent1, parent2):
        """Applies single-point crossover."""
        if self.num_sentences > 1 and np.random.rand() < self.p_crossover:
            point = np.random.randint(1, self.num_sentences)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2
        return parent1.copy(), parent2.copy()

    def _mutate(self, chromosome):
        """Applies mutation based on sequence bit inversion."""
        if np.random.rand() < self.p_mutation:
            if self.num_sentences > 1:
                idx1, idx2 = sorted(np.random.choice(self.num_sentences, 2, replace=False))
                chromosome[idx1:idx2] = 1 - chromosome[idx1:idx2]
            elif self.num_sentences == 1:
                chromosome[0] = 1 - chromosome[0]
        return chromosome

    def run(self, fitness_fn):
        """Runs the genetic algorithm using the specified fitness function."""
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


class GaSUMPipeline:
    def __init__(self, doc_text, target_len=3, model_name="bert-base-uncased"):
        self.doc_text = doc_text
        self.sentences = sent_tokenize(doc_text)
        self.target_len = target_len

        # Load BERT model and tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Precompute the document embedding r_D using the [CLS] token
        self.doc_repr = self._get_bert_representation(self.doc_text)
        
        # Cache dictionary to store representations of evaluated candidate summaries
        self.representation_cache = {}

    def _get_bert_representation(self, text):
        """Extracts the [CLS] token representation for the input text."""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512, 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Take the representation of the [CLS] token (index 0)
            cls_representation = outputs.last_hidden_state[:, 0, :]
        return cls_representation

    def _decode_summary(self, chromosome):
        """Converts a binary chromosome back into a text summary string."""
        selected_sentences = [self.sentences[i] for i, gene in enumerate(chromosome) if gene == 1]
        return " ".join(selected_sentences)

    def fitness_gasum(self, chromosome):
        """
        Computes fitness using cached representations to prevent redundant BERT passes.
        """
        selected_indices = np.where(chromosome == 1)[0]
        if len(selected_indices) == 0:
            return -100.0  # Constraint penalty for empty selections

        # Use tuple of indices as cache key
        cache_key = tuple(selected_indices)
        
        if cache_key in self.representation_cache:
            cand_repr = self.representation_cache[cache_key]
        else:
            candidate_text = self._decode_summary(chromosome)
            cand_repr = self._get_bert_representation(candidate_text)
            self.representation_cache[cache_key] = cand_repr
        
        # Compute cosine similarity between r_D and r_C
        cos_sim = torch.cosine_similarity(self.doc_repr, cand_repr, dim=1).item()
        
        # Length constraint/penalty to keep summary near target size
        length_penalty = abs(len(selected_indices) - self.target_len) * 0.1
        
        return cos_sim - length_penalty

    def generate_summary(self, pop_size=50, generations=10):
        if len(self.sentences) == 0:
            return ""
        
        ga = GeneticAlgorithmSummarizer(
            self.sentences, 
            pop_size=pop_size, 
            generations=generations, 
            target_len=self.target_len,
            tournament_size=32
        )
        
        best_chrom = ga.run(self.fitness_gasum)
        return self._decode_summary(best_chrom)


def compute_f1_score(candidate_summary, reference_summary):
    """Computes a token-overlap F1 score to evaluate overlap."""
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
    df = load_parquet_data()
    # Safely handle small datasets for evaluation
    num_available = len(df)
    start_idx = min(1000, max(0, num_available - 3))
    end_idx = min(start_idx + 3, num_available)
    ds = df.iloc[start_idx:end_idx].to_dict(orient="records")

    print("=== RUNNING GaSUM EVALUATION ===")
    num_to_evaluate = 3  
    target_length = 3    

    scores_gasum = []

    for idx, item in enumerate(ds):
        if idx >= num_to_evaluate:
            break

        doc_text = item.get('document', '')
        ref_text = item.get('abstract', '')

        sentences = sent_tokenize(doc_text)
        if len(sentences) <= target_length:
            print(f"\nSkipping Document {idx + 1} (Too few sentences: {len(sentences)})")
            continue

        print(f"\nProcessing Document {idx + 1}/{num_to_evaluate} ({len(sentences)} sentences)")
        
        pipeline = GaSUMPipeline(doc_text, target_len=target_length, model_name="bert-base-uncased")
        
        # Generation runs faster due to cached embeddings
        summary_gasum = pipeline.generate_summary(pop_size=20, generations=5)
        f1_gasum = compute_f1_score(summary_gasum, ref_text)
        scores_gasum.append(f1_gasum)

        print(f"Generated Summary: {summary_gasum[:150]}...")
        print(f"GaSUM F1 Score: {f1_gasum:.4f}")

    if scores_gasum:
        print("\n" + "="*50)
        print(f"EVALUATION COMPLETE")
        print("="*50)
        print(f"Average F-Measure (GaSUM): {np.mean(scores_gasum):.4f}")
        print("="*50)