1. ROUGE-1 (43.01%) — Solid

    What it means: Roughly 43% of the individual words in the reference abstract were successfully captured in your 4 extracted sentences.

    Context: On the arXiv dataset, state-of-the-art (SOTA) deep-learning models (such as LED, PEGASUS, or BART) generally achieve ROUGE-1 scores between 46.0% and 49.0%. For a simple, lightweight statistical algorithm running in seconds on a CPU, achieving 43.01% is highly competitive and indicates the parameter optimization successfully prioritized high-value content sentences.

2. ROUGE-2 (17.45%) — Strong for this Architecture

    What it means: Over 17% of consecutive word pairs (bigrams) matched the reference abstract exactly.

    Context: ROUGE-2 is usually much lower than ROUGE-1 because matching exact word pairs is significantly harder. Neural models typically score between 17.5% and 21.0% on arXiv. Your score of 17.45% is strong, showing that optimizing the bigram scoring and position weights successfully captured multi-word scientific terminology.

3. ROUGE-L (24.15%) — Moderate

    What it means: Measures the longest common subsequence of words between your summary and the abstract.

    Context: Extractive summaries often score lower on ROUGE-L when evaluating long documents. This is because extractive systems pull long, complete sentences from the source document, whereas the reference abstract consists of short, highly compressed sentences. The grammatical structure of an extracted sentence will naturally differ from a rewritten abstract sentence, keeping ROUGE-L in the low-to-mid 20s.

4. Cosine Similarity (62.09%) — High

    What it means: Measures overall vocabulary alignment in vector space.

    Context: A score of 62.09% indicates a strong thematic alignment between the vocabulary of your generated summary and the gold-standard abstract. This confirms that the model successfully stayed on-topic without drifting into irrelevant portions of the paper.