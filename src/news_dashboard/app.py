import os
import sys
from pathlib import Path

from flask import Flask, render_template, request, jsonify

APP_DIR = Path(__file__).resolve().parent
SRC_ROOT = APP_DIR.parent
for path in (str(SRC_ROOT), str(APP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from db.database import (  # noqa: E402
    distinct_topics,
    get_item_texts,
    search_today_items,
    sentiment_trend,
    top_topics,
)
import nlp_utils  # noqa: E402

app = Flask(__name__)


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
    return jsonify(distinct_topics())


@app.route("/api/sentiment-trend")
def api_sentiment_trend():
    topic = request.args.get("topic", "")
    days = int(request.args.get("days", 30))
    result = sentiment_trend(topic, days)
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
