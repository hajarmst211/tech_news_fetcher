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
from gensim.models import LdaMulticore, CoherenceModel
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases, Phraser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader import load_parquet_data

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
            continue
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tagged_tokens = nltk.pos_tag(tokens)
        cleaned_tokens = []
        for word, tag in tagged_tokens:
            if word not in stop_words:
                pos = get_wordnet_pos(tag)
                cleaned_tokens.append(lemmatizer.lemmatize(word, pos))
        processed_texts.append(cleaned_tokens)
    return processed_texts

def build_ngrams(texts):
    logging.info("Building ngrams")
    bigram_phrases = Phrases(texts, min_count=2, threshold=5)
    trigram_phrases = Phrases(bigram_phrases[texts], threshold=5)
    bigram_mod = Phraser(bigram_phrases)
    trigram_mod = Phraser(trigram_phrases)
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def main():
    logging.info("Loading parquet data")
    df = load_parquet_data()
    text_column = next((col for col in df.columns if col.lower() in ['text', 'content', 'body']), df.columns[0])
    
    logging.info("Selecting documents from dataset")
    raw_texts = df[text_column].dropna().tolist()
    
    processed_tokens = preprocess_text(raw_texts)
    ngram_tokens = build_ngrams(processed_tokens)
    
    logging.info("Creating dictionary and corpus")
    dictionary = Dictionary(ngram_tokens)
    dictionary.filter_extremes(no_below=1, no_above=1.0)
    corpus = [dictionary.doc2bow(text) for text in ngram_tokens]
    
    num_topics = 3
    passes = 20
    
    logging.info("Training LDA model on the dataset")
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        workers=max(1, os.cpu_count() - 1),
        random_state=42
    )
    
    logging.info("Retrieving learned topics from the dataset")
    print("\n=== LEARNED TOPICS FROM THE DATASET ===")
    topics = lda_model.print_topics(num_topics=num_topics, num_words=5)
    for topic_id, words in topics:
        print(f"Topic {topic_id}: {words}")
    print("=======================================\n")
    
    logging.info("Calculating per-document measurements")
    print("=== PER-DOCUMENT ANALYSIS ===")
    for i, doc_bow in enumerate(corpus):
        topic_distribution = lda_model[doc_bow]
        dominant_topic = sorted(topic_distribution, key=lambda x: x[1], reverse=True)[0]
        
        doc_perplexity = lda_model.log_perplexity([doc_bow])
        
        print(f"\nDocument {i+1}:")
        print(f"  Topic Distribution: {topic_distribution}")
        print(f"  Suggested Topic: Topic {dominant_topic[0]} (Probability Score: {dominant_topic[1]:.4f})")
        print(f"  Document-Specific Perplexity contribution: {doc_perplexity:.4f}")
    print("=============================")

if __name__ == '__main__':
    main()