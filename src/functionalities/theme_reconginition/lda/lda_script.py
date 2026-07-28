import sys
import os
import re
import logging
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from gensim.models import LdaMulticore
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases, Phraser
from gensim.models import CoherenceModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'cs_papers_api.csv')
OUTPUT_MD_PATH = os.path.join(SCRIPT_DIR, 'lda_results.md')

sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..')))
from data_loader import load_data

logging.info("Downloading NLTK resources")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(texts):
    logging.info(f"Preprocessing {len(texts)} documents")
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    processed_texts = []
    for text in texts:
        if not isinstance(text, str):
            processed_texts.append([])
            continue
        tokens = word_tokenize(text)
        tagged_tokens = nltk.pos_tag(tokens)
        cleaned_tokens = []
        for word, tag in tagged_tokens:
            word_clean = re.sub(r'[^a-zA-Z]', '', word).lower()
            if len(word_clean) > 2 and word_clean not in stop_words:
                pos = get_wordnet_pos(tag)
                cleaned_tokens.append(lemmatizer.lemmatize(word_clean, pos))
        processed_texts.append(cleaned_tokens)
    return processed_texts

def build_ngrams(texts):
    logging.info("Building ngrams")
    bigram_phrases = Phrases(texts, min_count=2, threshold=5)
    trigram_phrases = Phrases(bigram_phrases[texts], threshold=5)
    bigram_mod = Phraser(bigram_phrases)
    trigram_mod = Phraser(trigram_phrases)
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def find_best_topic_count(corpus, dictionary, ngram_tokens, start=2, end=8):
    """
    Evaluates LDA models over a range of topic counts and returns the 
    number of topics that yields the highest coherence score.
    """
    logging.info(f"Evaluating topic models to find the optimal count between {start} and {end}...")
    best_coherence = -1
    best_k = start
    coherence_records = {}
    
    for k in range(start, end + 1):
        logging.info(f"Training evaluation model with k={k} topics...")
        temp_model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            passes=8,
            workers=max(1, os.cpu_count() - 1),
            random_state=42
        )
        coherence_model = CoherenceModel(
            model=temp_model, 
            texts=ngram_tokens, 
            dictionary=dictionary, 
            coherence='c_v'
        )
        score = coherence_model.get_coherence()
        coherence_records[k] = score
        logging.info(f"Result: k={k} achieved a coherence score of {score:.4f}")
        
        if score > best_coherence:
            best_coherence = score
            best_k = k
            
    return best_k, coherence_records

def main():
    logging.info("Loading parquet data")
    df = load_data(DATA_PATH).head(1000)
    
    text_column = next((col for col in df.columns if col.lower() in ['text', 'content', 'body']), df.columns[0])
    label_column = next((col for col in df.columns if col.lower() in ['label', 'category', 'target']), None)
    
    df_clean = df.dropna(subset=[text_column]).copy()
    raw_texts = df_clean[text_column].tolist()
    
    if label_column and label_column in df_clean.columns:
        labels = df_clean[label_column].tolist()
    else:
        labels = ["No Label Available"] * len(raw_texts)
        
    processed_tokens = preprocess_text(raw_texts)
    ngram_tokens = build_ngrams(processed_tokens)
    
    logging.info("Creating dictionary and corpus")
    dictionary = Dictionary(ngram_tokens)
    dictionary.filter_extremes(no_below=2, no_above=0.7)
    corpus = [dictionary.doc2bow(text) for text in ngram_tokens]
    
    optimal_num_topics, coherence_history = find_best_topic_count(corpus, dictionary, ngram_tokens, start=2, end=8)
    logging.info(f"Optimal number of topics determined: {optimal_num_topics}")
    
    passes = 20
    logging.info(f"Training final LDA model with {optimal_num_topics} topics...")
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=optimal_num_topics,
        passes=passes,
        workers=max(1, os.cpu_count() - 1),
        random_state=42
    )
    
    topic_strings = {}
    for topic_id in range(optimal_num_topics):
        words = lda_model.show_topic(topic_id, topn=5)
        word_list = [word for word, prop in words]
        topic_strings[topic_id] = f"Topic {topic_id} ({', '.join(word_list)})"

    logging.info("Assigning documents to optimal topics")
    document_entries = []
    
    for i, doc_bow in enumerate(corpus):
        print(f"Classifying document {i + 1}/{len(corpus)}", end='\r')
        
        doc_label = labels[i]
        if not doc_bow:
            document_entries.append((i + 1, doc_label, "None (Empty Text)"))
            continue
            
        topic_distribution = lda_model[doc_bow]
        dominant_topic = sorted(topic_distribution, key=lambda x: x[1], reverse=True)[0]
        dominant_topic_id = dominant_topic[0]
        
        suggested_topic_str = topic_strings[dominant_topic_id]
        document_entries.append((i + 1, doc_label, suggested_topic_str))
        
    print("\nDocument processing complete.")

    perplexity = lda_model.log_perplexity(corpus)
    coherence_model = CoherenceModel(model=lda_model, texts=ngram_tokens, dictionary=dictionary, coherence='c_v')
    final_coherence_score = coherence_model.get_coherence()

    logging.info(f"Writing output to {OUTPUT_MD_PATH}")
    with open(OUTPUT_MD_PATH, 'w', encoding='utf-8') as f:
        f.write("# Dynamic LDA Topic Modeling Report\n\n")
        
        f.write("## Topic Number Optimization\n")
        f.write("The model evaluated different topic counts to determine which configuration produced the most coherent word clusters:\n\n")
        f.write("| Number of Topics (k) | Coherence Score (C_V) | Status |\n")
        f.write("| --- | --- | --- |\n")
        for k, score in coherence_history.items():
            status = "**Selected** (Highest Coherence)" if k == optimal_num_topics else "Evaluated"
            f.write(f"| {k} | {score:.4f} | {status} |\n")
        f.write("\n")
        
        f.write("## Final Model Global Metrics\n")
        f.write("| Metric | Value | Description |\n")
        f.write("| --- | --- | --- |\n")
        f.write(f"| **Selected Topics Count** | {optimal_num_topics} | Number of dynamically determined categories |\n")
        f.write(f"| **Perplexity** | {perplexity:.4f} | Predictability score (lower represents a better mathematical fit) |\n")
        f.write(f"| **Coherence Score (C_V)** | {final_coherence_score:.4f} | Topic clarity score (higher is better) |\n\n")
        
        f.write("## Generated Topic Definitions\n")
        f.write("These topics represent the most coherent word patterns found across your files:\n\n")
        for topic_id, topic_str in topic_strings.items():
            f.write(f"- **Topic {topic_id}**: `{topic_str}`\n")
        f.write("\n")
        
        f.write("## Document Classification List\n")
        f.write("Below is each processed document accompanied by its dataset category and its mathematically predicted topic:\n\n")
        f.write("| Document ID | Original Dataset Label | Predicted Topic |\n")
        f.write("| --- | --- | --- |\n")
        for doc_id, label, suggested_topic in document_entries:
            f.write(f"| {doc_id} | {label} | {suggested_topic} |\n")

    logging.info("Execution complete")

if __name__ == '__main__':
    main()