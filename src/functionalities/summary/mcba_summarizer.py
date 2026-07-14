import random
import numpy as np
from `sklearn`.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download('punkt', quiet=True)

class MCBAGASummarizer:
    def __init__(self, pop_size=20, generations=30, crossover_rate=0.8, mutation_rate=0.1):
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def _get_sentence_positions(self, num_sentences):
        return [1.0 / (i + 1) for i in range(num_sentences)]

    def _calculate_fitness(self, chromosome, tfidf_matrix, position_scores, target_length):
        selected_indices = [i for i, gene in enumerate(chromosome) if gene == 1]
        if not selected_indices:
            return 0.0
        
        penalty = abs(len(selected_indices) - target_length) / float(target_length)
        
        selected_vectors = tfidf_matrix[selected_indices]
        centroid = tfidf_matrix.mean(axis=0)
        coverage = cosine_similarity(selected_vectors, centroid).mean()
        
        if len(selected_indices) > 1:
            pairwise_sim = cosine_similarity(selected_vectors)
            redundancy = (pairwise_sim.sum() - len(selected_indices)) / (len(selected_indices) * (len(selected_indices) - 1))
        else:
            redundancy = 0.0
            
        pos_score = sum(position_scores[i] for i in selected_indices) / len(selected_indices)
        
        fitness = (coverage - redundancy + pos_score) - (0.5 * penalty)
        return max(0.0, fitness)

    def _crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            point = random.randint(1, len(parent1) - 1)
            return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
        return parent1.copy(), parent2.copy()

    def _mutate(self, chromosome):
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]
        return chromosome

    def summarize(self, text, target_length=3):
        sentences = nltk.sent_tokenize(text)
        num_sentences = len(sentences)
        
        if num_sentences <= target_length:
            return sentences

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        position_scores = self._get_sentence_positions(num_sentences)

        population = []
        for _ in range(self.pop_size):
            chromosome = [0] * num_sentences
            indices = random.sample(range(num_sentences), target_length)
            for idx in indices:
                chromosome[idx] = 1
            population.append(chromosome)

        for _ in range(self.generations):
            fitness_scores = [self._calculate_fitness(c, tfidf_matrix, position_scores, target_length) for c in population]
            total_fitness = sum(fitness_scores)
            
            if total_fitness == 0:
                selection_probs = [1.0 / self.pop_size] * self.pop_size
            else:
                selection_probs = [f / total_fitness for f in fitness_scores]

            new_population = []
            for _ in range(self.pop_size // 2):
                parent1 = population[np.random.choice(self.pop_size, p=selection_probs)]
                parent2 = population[np.random.choice(self.pop_size, p=selection_probs)]
                
                child1, child2 = self._crossover(parent1, parent2)
                new_population.append(self._mutate(child1))
                new_population.append(self._mutate(child2))
            
            population = new_population

        final_fitness = [self._calculate_fitness(c, tfidf_matrix, position_scores, target_length) for c in population]
        best_chromosome = population[np.argmax(final_fitness)]
        
        summary_sentences = [sentences[i] for i, gene in enumerate(best_chromosome) if gene == 1]
        return summary_sentences

def calculate_f_measure(evaluated_summary, reference_summary):
    eval_words = set(nltk.word_tokenize(" ".join(evaluated_summary).lower()))
    ref_words = set(nltk.word_tokenize(" ".join(reference_summary).lower()))
    
    if not eval_words or not ref_words:
        return 0.0
        
    intersection = eval_words.intersection(ref_words)
    precision = len(intersection) / len(eval_words)
    recall = len(intersection) / len(ref_words)
    
    if precision + recall == 0:
        return 0.0
        
    return 2 * (precision * recall) / (precision + recall)

if __name__ == "__main__":
    document = (
        "The political landscape is shifting rapidly. "
        "Economic factors are driving policy adjustments. "
        "Citizens are demanding more transparency. "
        "Government officials are responding with new initiatives. "
        "Global alliances are playing a significant role. "
        "Future projections remain highly uncertain."
    )
    reference = [
        "The political landscape is shifting rapidly.",
        "Economic factors are driving policy adjustments.",
        "Citizens are demanding more transparency."
    ]

    summarizer = MCBAGASummarizer()
    generated = summarizer.summarize(document, target_length=3)
    f_measure = calculate_f_measure(generated, reference)

    print("Generated Summary:")
    for sent in generated:
        print("-", sent)
    print(f"\nF-measure: {f_measure:.4f}")