import os
import sys
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util

APP_DIR = Path(__file__).resolve().parent
SRC_ROOT = APP_DIR.parent
for path in (str(SRC_ROOT), str(APP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from db.database import (
    distinct_topics,
    get_item_texts,
    search_today_items,
    sentiment_trend,
    top_topics,
)
import nlp_utils

app = Flask(__name__)

_canonical_to_raw = {}
_raw_to_canonical = {}


def cluster_topics(raw_topics, threshold=0.55):
    if not raw_topics:
        return {}, {}
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(raw_topics, convert_to_tensor=True)
    assigned = set()
    canonical_to_raw = {}
    raw_to_canonical = {}
    for idx, topic in enumerate(raw_topics):
        if topic in assigned:
            continue
        cos_scores = util.cos_sim(embeddings[idx], embeddings)[0]
        similar_indices = [i for i, score in enumerate(cos_scores) if score >= threshold]
        cluster_members = []
        for i in similar_indices:
            member = raw_topics[i]
            if member not in assigned:
                cluster_members.append(member)
                assigned.add(member)
        if cluster_members:
            canonical = min(cluster_members, key=len)
            canonical_to_raw[canonical] = cluster_members
            for member in cluster_members:
                raw_to_canonical[member] = canonical
    return canonical_to_raw, raw_to_canonical


def get_or_build_topic_clusters():
    global _canonical_to_raw, _raw_to_canonical
    if not _canonical_to_raw:
        raw_topics = distinct_topics()
        _canonical_to_raw, _raw_to_canonical = cluster_topics(raw_topics, threshold=0.55)
    return _canonical_to_raw, _raw_to_canonical


@app.route("/")
def feed():
    return render_template("feed.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/feed")
def api_feed():
    query = request.args.get("q", "").strip()
    items = search_today_items(query)
    return jsonify(items)


@app.route("/api/ngrams")
def api_ngrams():
    n = int(request.args.get("n", 1))
    days = int(request.args.get("days", 30))
    top_k = int(request.args.get("top_k", 12))
    if n not in (1, 2, 3, 4):
        return jsonify({"error": "n must be between 1 and 4"}), 400
    texts = get_item_texts(days)
    result = nlp_utils.top_ngrams(texts, n=n, top_k=top_k)
    return jsonify(result)


@app.route("/api/topics")
def api_topics():
    days = int(request.args.get("days", 30))
    limit = int(request.args.get("limit", 15))
    result = top_topics(days, limit)
    return jsonify(result)


@app.route("/api/topics/list")
def api_topics_list():
    canonical_map, _ = get_or_build_topic_clusters()
    return jsonify(sorted(list(canonical_map.keys())))


@app.route("/api/sentiment-trend")
def api_sentiment_trend():
    topic = request.args.get("topic", "")
    days = int(request.args.get("days", 30))
    canonical_map, _ = get_or_build_topic_clusters()
    raw_topics = canonical_map.get(topic, [topic] if topic else [])
    result = sentiment_trend(raw_topics, days)
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)