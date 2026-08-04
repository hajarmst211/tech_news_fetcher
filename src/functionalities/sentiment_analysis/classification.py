import os
import time
import numpy as np
from data_loader import load_data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

start_time = time.time()

print("Step 1: Loading data from data_loader...")
df = load_data()
print(f"Data loaded. Total rows: {len(df)}")

X = df['text']
y = df['label']

print("Step 2: Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

print("Step 3: Vectorizing text data using TF-IDF...")
vector_start = time.time()
vectorizer = TfidfVectorizer(max_features=50000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"Vectorization complete. Vocabulary size: {X_train_vec.shape[1]} (took {time.time() - vector_start:.2f} seconds)")

models = {
    "Naive Bayes": MultinomialNB(),
    "SVM (LinearSVC)": LinearSVC( max_iter=5000),
    "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=42)
}

results = {}
all_predictions = {}
all_confidences = {}

def get_confidences(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return np.max(proba, axis=1) * 100
    decision = model.decision_function(X)
    if decision.ndim == 1:
        proba = 1 / (1 + np.exp(-decision))
        return np.where(proba > 0.5, proba, 1 - proba) * 100
    exp_scores = np.exp(decision - np.max(decision, axis=1, keepdims=True))
    proba = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return np.max(proba, axis=1) * 100

print("Step 4: Training and evaluating models...")
for name, model in models.items():
    print(f"Starting {name} training...")
    model_start = time.time()
    model.fit(X_train_vec, y_train)
    print(f"Finished training {name} in {time.time() - model_start:.2f} seconds.")
    
    print(f"Evaluating {name}...")
    predictions = model.predict(X_test_vec)
    all_predictions[name] = predictions
    
    print(f"Computing confidence scores for {name}...")
    confidences = get_confidences(model, X_test_vec)
    all_confidences[name] = confidences
    
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    
    classes = sorted(list(set(y_test)))
    cm = confusion_matrix(y_test, predictions, labels=classes)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    results[name] = {
        "accuracy": accuracy,
        "report": report,
        "classes": classes,
        "cm_percent": cm_percent
    }
    print(f"Finished evaluating {name}.")

print("Step 5: Writing results to markdown file...")
script_dir = os.path.dirname(os.path.abspath(__file__))
markdown_path = os.path.join(script_dir, "classification_results.md")

with open(markdown_path, "w", encoding="utf-8") as f:
    f.write("# Sentiment Analysis Evaluation Results\n\n")
    
    f.write("## 1. Overall Model Performance\n\n")
    f.write("| Model | Accuracy |\n")
    f.write("|---|---|\n")
    for name, metrics in results.items():
        f.write(f"| {name} | {metrics['accuracy']:.4f} |\n")
    f.write("\n")
    
    for name, metrics in results.items():
        f.write(f"### Detailed Report: {name}\n\n")
        f.write("| Class | Precision | Recall | F1-Score |\n")
        f.write("|---|---|---|---|\n")
        for label, score in metrics['report'].items():
            if isinstance(score, dict):
                f.write(f"| {label} | {score['precision']:.4f} | {score['recall']:.4f} | {score['f1-score']:.4f} |\n")
        f.write("\n")
        
    f.write("## 2. Prediction Distribution Matrices (Actual vs Predicted %)\n\n")
    for name, metrics in results.items():
        f.write(f"### {name} - Transition Percentages\n")
        classes = metrics['classes']
        headers = " | ".join([f"Predicted {c} (%)" for c in classes])
        f.write(f"| Actual Label | {headers} |\n")
        f.write("|" + "---|"* (len(classes) + 1) + "\n")
        
        for i, actual_class in enumerate(classes):
            row_vals = " | ".join([f"{metrics['cm_percent'][i, j]:.2f}%" for j in range(len(classes))])
            f.write(f"| {actual_class} | {row_vals} |\n")
        f.write("\n")
        
    f.write("## 3. Comparative Prediction Samples (First 100 Samples)\n\n")
    f.write("| Sample # | Text | Actual Label | Naive Bayes | SVM (LinearSVC) | Decision Tree |\n")
    f.write("|---|---|---|---|---|---|\n")
    
    test_texts = list(X_test)
    test_labels = list(y_test)
    limit = min(100, len(test_texts))
    
    for idx in range(limit):
        clean_text = str(test_texts[idx]).replace("\n", " ").replace("|", "\\|")
        actual = test_labels[idx]
        nb_pred = all_predictions["Naive Bayes"][idx]
        svm_pred = all_predictions["SVM (LinearSVC)"][idx]
        dt_pred = all_predictions["Decision Tree"][idx]
        nb_conf = all_confidences["Naive Bayes"][idx]
        svm_conf = all_confidences["SVM (LinearSVC)"][idx]
        dt_conf = all_confidences["Decision Tree"][idx]
        f.write(f"| {idx + 1} | {clean_text} | {actual} | {nb_pred} ({nb_conf:.1f}%) | {svm_pred} ({svm_conf:.1f}%) | {dt_pred} ({dt_conf:.1f}%) |\n")

print(f"Results successfully saved to: {markdown_path}")
print(f"Total pipeline execution time: {time.time() - start_time:.2f} seconds")