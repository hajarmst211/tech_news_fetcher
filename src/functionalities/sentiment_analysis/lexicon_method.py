import os
import sys
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data_loader import load_data

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    return None

def compute_confidence(avg_score, label):
    if label == "positive":
        conf = (avg_score - 0.15) / (1.0 - 0.15)
    elif label == "negative":
        conf = (-0.05 - avg_score) / (1.0 - 0.05)
    else:
        conf = 1.0 - min(1.0, abs(avg_score - 0.05) / 0.1)
    return max(0.0, min(1.0, conf)) * 100

def evaluate_sentiment(text, lemmatizer, acronyms, emoticons, contextual_words):
    text = str(text).lower()
    for phrase, replacement in contextual_words.items():
        text = text.replace(phrase, replacement)
    
    words = text.split()
    processed_words = []
    for word in words:
        if word in emoticons:
            processed_words.append(emoticons[word])
        elif word in acronyms:
            processed_words.extend(acronyms[word].split())
        else:
            processed_words.append(word)
            
    reconstructed_text = " ".join(processed_words)
    tokens = word_tokenize(reconstructed_text)
    tagged_tokens = nltk.pos_tag(tokens)
    
    score = 0.0
    count = 0
    
    for word, tag in tagged_tokens:
        wn_tag = get_wordnet_pos(tag)
        if wn_tag:
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            synsets = list(swn.senti_synsets(lemma, wn_tag))
            if synsets:
                senti_syn = synsets[0]
                net_score = senti_syn.pos_score() - senti_syn.neg_score()
                if net_score != 0.0:
                    score += net_score
                    count += 1
                    
    if count == 0:
        return "neutral", 0.0
        
    avg_score = score / count
    
    if avg_score > 0.15:
        label = "positive"
    elif avg_score < -0.05:
        label = "negative"
    else:
        label = "neutral"
    
    return label, compute_confidence(avg_score, label)

def normalize_label(label):
    label_str = str(label).strip().lower()
    if label_str in ["positive", "pos", "1", "positive"]:
        return "positive"
    elif label_str in ["negative", "neg", "0", "negative"]:
        return "negative"
    else:
        return "neutral"

def run_pipeline(max_samples=1000):
    print("Execution State: Initiating download of NLTK dependencies...")
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('sentiwordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("Execution State: NLTK dependencies ready.")

    print("Execution State: Loading raw data via data_loader...")
    df = load_data()
    total_raw_rows = len(df)
    print(f"Execution State: Raw dataset loaded with {total_raw_rows} records.")
    
    if max_samples is not None and total_raw_rows > max_samples:
        print(f"Execution State: Sampling {max_samples} random records from the dataset...")
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"Execution State: Working dataset constrained to {len(df)} records.")
    else:
        print(f"Execution State: Using all {total_raw_rows} records for evaluation.")

    lemmatizer = WordNetLemmatizer()
    
    acronyms = {
        "tia": "thank you in advance",
        "lol": "laughing out loud",
        "gr8": "great",
        "omg": "oh my god",
        "asap": "as soon as possible",
        "fyi": "for your information"
    }

    emoticons = {
        ":)": "happy",
        ":-)": "happy",
        ":D": "happy",
        "😀": "happy",
        ":(": "sad",
        ":-(": "sad",
        "😢": "sad"
    }

    contextual_words = {
        "not good": "bad",
        "not bad": "good",
        "no delay": "fast"
    }
    
    print("Execution State: Normalizing ground truth labels...")
    y_true = [normalize_label(label) for label in df['label']]
    y_pred = []
    samples_data = []
    
    total_records = len(df)
    print(f"Execution State: Commencing sentiment evaluation of {total_records} records...")
    
    for idx, text in enumerate(df['text']):
        prediction, confidence = evaluate_sentiment(text, lemmatizer, acronyms, emoticons, contextual_words)
        y_pred.append(prediction)
        
        if len(samples_data) < 100:
            samples_data.append({
                "text": str(text),
                "actual": y_true[idx],
                "predicted": prediction,
                "confidence": confidence
            })
        
        if (idx + 1) % max(1, total_records // 10) == 0 or (idx + 1) == total_records:
            progress_pct = ((idx + 1) / total_records) * 100
            print(f"Execution State: Evaluated {idx + 1}/{total_records} records ({progress_pct:.1f}% complete)...")
            
    print("Execution State: Computation of predictions finalized.")
    print("Execution State: Calculating system performance metrics...")
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    print("Execution State: Computing prediction combination distribution...")
    actual_counts = {"positive": 0, "negative": 0, "neutral": 0}
    matrix = {
        ("positive", "positive"): 0, ("positive", "negative"): 0, ("positive", "neutral"): 0,
        ("negative", "positive"): 0, ("negative", "negative"): 0, ("negative", "neutral"): 0,
        ("neutral", "positive"): 0, ("neutral", "negative"): 0, ("neutral", "neutral"): 0
    }
    
    for act, pred in zip(y_true, y_pred):
        actual_counts[act] += 1
        matrix[(act, pred)] += 1
        
    def safe_percentage(act, pred):
        total = actual_counts[act]
        if total == 0:
            return 0.0
        return (matrix[(act, pred)] / total) * 100

    print(f"Execution State: Metrics calculated. Accuracy: {accuracy:.4f}, F1: {f1:.4f}.")
    
    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_directory, "evaluation_results.md")
    
    print(f"Execution State: Writing results, metrics matrix, and sample details to '{output_path}'...")
    
    markdown_content = f"""# Senti_Con_Acron Evaluation Report

## Configuration
*   **Total Evaluated Samples:** {len(df)}

## Performance Metrics
| Metric | Score (%) |
| :--- | :--- |
| **Precision** | {precision * 100:.2f}% |
| **Recall** | {recall * 100:.2f}% |
| **F-Measure** | {f1 * 100:.2f}% |
| **Accuracy** | {accuracy * 100:.2f}% |

## Prediction Distribution Matrix (Row-Normalized)
The table below details how actual classes were distributed across the predicted classes. Each row sums up to approximately 100%.

| Actual Class (Count) | Predicted Positive (%) | Predicted Negative (%) | Predicted Neutral (%) |
| :--- | :--- | :--- | :--- |
| **Positive** ({actual_counts['positive']}) | {safe_percentage('positive', 'positive'):.2f}% | {safe_percentage('positive', 'negative'):.2f}% | {safe_percentage('positive', 'neutral'):.2f}% |
| **Negative** ({actual_counts['negative']}) | {safe_percentage('negative', 'positive'):.2f}% | {safe_percentage('negative', 'negative'):.2f}% | {safe_percentage('negative', 'neutral'):.2f}% |
| **Neutral** ({actual_counts['neutral']}) | {safe_percentage('neutral', 'positive'):.2f}% | {safe_percentage('neutral', 'negative'):.2f}% | {safe_percentage('neutral', 'neutral'):.2f}% |

## Evaluation Sample Table (Subset of 100 Records)
| Index | Text | Actual Label | Predicted Label | Confidence (%) |
| :--- | :--- | :--- | :--- | :--- |
"""
    
    for idx, sample in enumerate(samples_data):
        sanitized_text = sample["text"].replace("\n", " ").replace("|", "\\|")
        markdown_content += f"| {idx + 1} | {sanitized_text} | {sample['actual']} | {sample['predicted']} | {sample['confidence']:.2f}% |\n"
        
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        
    print("Execution State: Pipeline run complete. Markdown file generated.")

if __name__ == "__main__":
    run_pipeline(max_samples=5000)