import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]
SCORE_MAP = {"Negative": -1.0, "Neutral": 0.0, "Positive": 1.0}


def _sentiment_score(label: str) -> float:
    return SCORE_MAP.get(label, 0.0)


def _label_from_mean(mean_score: float) -> str:
    if mean_score > 0.33:
        return "Positive"
    if mean_score < -0.33:
        return "Negative"
    return "Neutral"


class SVMClassifier:
    def __init__(self, max_features=50000, max_iter=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = LinearSVC(max_iter=max_iter)
        self.classes_ = None

    def train(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.classes_ = sorted(self.model.classes_)
        return self

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        return list(self.model.predict(X))

    def predict_with_confidence(self, texts):
        X = self.vectorizer.transform(texts)
        labels = self.model.predict(X)
        decision = self.model.decision_function(X)
        if decision.ndim == 1:
            conf = np.abs(decision) * 100
        else:
            conf = np.max(decision, axis=1) * 100
        return list(zip(labels, conf))

    def analyze(self, comments):
        predictions = np.array(self.predict(comments))
        scores = np.array([_sentiment_score(p) for p in predictions])
        counts = {c: int(np.sum(predictions == c)) for c in SENTIMENT_ORDER}
        total = len(predictions)
        distribution = {c: counts[c] / total if total else 0.0 for c in SENTIMENT_ORDER}
        mean_score = float(scores.mean()) if total else 0.0
        return {
            "total_comments": total,
            "predictions": predictions,
            "counts": counts,
            "distribution": distribution,
            "mean_score": mean_score,
            "overall_sentiment": _label_from_mean(mean_score),
        }


def _load_pretrained(path=None):
    if path and os.path.exists(path):
        import joblib
        return joblib.load(path)
    return None


def _load_training_data():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from data_loader import load_data
    return load_data()


def train_classifier(max_features=50000, max_iter=5000, cache_path=None):
    start = time.time()
    print("Loading training data...")
    df = _load_training_data()
    print(f"Loaded {len(df)} comments.")

    clf = SVMClassifier(max_features=max_features, max_iter=max_iter)
    print("Training SVM...")
    clf.train(df["text"], df["label"])
    print(f"Training complete in {time.time() - start:.2f}s.")

    if cache_path:
        import joblib
        try:
            joblib.dump(clf, cache_path)
            print(f"SVM model cached to {cache_path}")
        except Exception as e:
            print(f"  [WARN] Could not cache SVM model: {e}")

    return clf


def get_classifier(cache_path=None):
    pretrained = _load_pretrained(cache_path)
    if pretrained is not None:
        print("Loaded cached SVM model.")
        return pretrained
    return train_classifier(cache_path=cache_path)


def main():
    clf = get_classifier()

    comments = sys.stdin.read().splitlines() if not sys.stdin.isatty() else None

    if comments is None:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "comments.csv")
        df_in = pd.read_csv(csv_path, usecols=["CommentText"])
        comments = df_in["CommentText"].dropna().tolist()

    comments = [c for c in comments if str(c).strip()]
    result = clf.analyze(comments)

    print(f"\nAnalyzed {result['total_comments']} comments.")
    for label in SENTIMENT_ORDER:
        print(f"  {label}: {result['counts'][label]} ({result['distribution'][label] * 100:.2f}%)")
    print(f"Overall sentiment (mean score {result['mean_score']:.4f}): {result['overall_sentiment']}")


if __name__ == "__main__":
    main()
