import sys
import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from gensim.models import LdaMulticore, CoherenceModel
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases, Phraser
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader import load_parquet_data

print("Downloading NLTK resources")
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
    print("Building ngrams")
    bigram_phrases = Phrases(texts, min_count=2, threshold=5)
    trigram_phrases = Phrases(bigram_phrases[texts], threshold=5)
    bigram_mod = Phraser(bigram_phrases)
    trigram_mod = Phraser(trigram_phrases)
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lda_grid_search(corpus, dictionary, texts, topic_range, pass_range):
    best_coherence = -1
    best_params = {}
    results = []
    for num_topics in topic_range:
        for passes in pass_range:
            lda_model = LdaMulticore(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                passes=passes,
                workers=max(1, os.cpu_count() - 1),
                random_state=42
            )
            coherence_model = CoherenceModel(
                model=lda_model,
                texts=texts,
                dictionary=dictionary,
                coherence='c_v'
            )
            coherence_score = coherence_model.get_coherence()
            results.append({
                'num_topics': num_topics,
                'passes': passes,
                'coherence': coherence_score
            })
            if coherence_score > best_coherence:
                best_coherence = coherence_score
                best_params = {'num_topics': num_topics, 'passes': passes}
    return best_params, best_coherence, pd.DataFrame(results)

def plsa_grid_search(texts, topic_range):
    best_coherence = -1
    best_params = {}
    results = []
    joined_texts = [' '.join(doc) for doc in texts]
    vectorizer = CountVectorizer(max_df=0.95, min_df=2)
    dtm = vectorizer.fit_transform(joined_texts)
    feature_names = vectorizer.get_feature_names_out()
    for num_topics in topic_range:
        nmf = NMF(
            n_components=num_topics,
            beta_loss='kullback-leibler',
            solver='mu',
            max_iter=200,
            random_state=42
        )
        nmf.fit(dtm)
        topics = []
        for topic_idx, topic in enumerate(nmf.components_):
            top_features_ind = topic.argsort()[:-11:-1]
            topics.append([feature_names[i] for i in top_features_ind])
        coherence_model = CoherenceModel(
            topics=topics,
            texts=texts,
            dictionary=Dictionary(texts),
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        results.append({
            'num_topics': num_topics,
            'coherence': coherence_score
        })
        if coherence_score > best_coherence:
            best_coherence = coherence_score
            best_params = {'num_topics': num_topics}
    return best_params, best_coherence, pd.DataFrame(results)

def main():
    df = load_parquet_data()
    text_column = next((col for col in df.columns if col.lower() in ['text', 'content', 'body']), df.columns[0])
    raw_texts = df[text_column].dropna().head(5).tolist()
    processed_tokens = preprocess_text(raw_texts)
    ngram_tokens = build_ngrams(processed_tokens)
    dictionary = Dictionary(ngram_tokens)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in ngram_tokens]
    topic_range = [3, 5, 10]
    pass_range = [5, 10, 20]
    best_lda, best_lda_coh, lda_df = lda_grid_search(corpus, dictionary, ngram_tokens, topic_range, pass_range)
    print("Best LDA Parameters:", best_lda)
    print("Best LDA Coherence:", best_lda_coh)
    best_plsa, best_plsa_coh, plsa_df = plsa_grid_search(ngram_tokens, topic_range)
    print("Best PLSA Parameters:", best_plsa)
    print("Best PLSA Coherence:", best_plsa_coh)

if __name__ == '__main__':
    main()