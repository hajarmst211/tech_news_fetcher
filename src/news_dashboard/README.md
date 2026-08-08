# signal — interface for the news pipeline

Two pages:

- `/` feed — search bar over today's fetched items, each card shows summary, top 3 topics, comment sentiment (only shown when the item has comments), and a link.
- `/dashboard` — three charts:
  - n-grams (n = 1 to 4), ranked by TF-IDF weight, with near-duplicate n-grams removed by cosine similarity between candidate term vectors.
  - most frequent topics, from `items.topics`.
  - sentiment / trend by topic, built from `comments.sentiment_label` joined against `items.topics`, filterable by topic.

## setup

```
cd src/news_dashboard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000

## notes

- The feed only searches items where `fetched_at::date = CURRENT_DATE` (any run today), as requested — not a live fetch triggered by the search itself.
- The n-gram and topic charts default to a 30 day window (`?days=` query param on the API routes if you want to change it).
- The DB helpers live in `src/db/database.py` and read `DATABASE_URL` from the project root `.env` via `dotenv`, matching your existing pipeline setup.
