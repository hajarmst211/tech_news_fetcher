import os
import time
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import DictCursor, execute_values
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
env_path = PROJECT_ROOT / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

db_url = os.environ.get("DATABASE_URL")
if not db_url:
    raise ValueError("Please set the DATABASE_URL environment variable.")

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
BATCH_SIZE = 50  

print("Initializing quantized model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
raw_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

quantized_model = torch.quantization.quantize_dynamic(
    raw_model, 
    {torch.nn.Linear}, 
    dtype=torch.qint8
)

sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model=quantized_model, 
    tokenizer=tokenizer, 
    device=-1  
)

LABEL_MAPPING = {
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative"
}

def fetch_comments_needing_sentiment(conn) -> list:
    query = """
        SELECT id, body_text 
        FROM comments 
        WHERE body_text IS NOT NULL 
          AND TRIM(body_text) != ''
          AND sentiment_label IS NULL
        ORDER BY id ASC;
    """
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(query)
        return [dict(row) for row in cur.fetchall()]

def update_comment_sentiments(conn, updates: list):
    query = """
        UPDATE comments AS c
        SET 
            sentiment_label = v.label::sentiment_label_enum,
            sentiment_score = v.score
        FROM (VALUES %s) AS v(label, score, id)
        WHERE c.id = v.id::bigint;
    """
    with conn.cursor() as cur:
        execute_values(cur, query, updates)
    conn.commit()

def process_batch(batch: list) -> list:
    updates = []
    for item in batch:
        comment_id = item["id"]
        text = item["body_text"]
        
        truncated_text = text[:1500] if text else ""
        
        if not truncated_text.strip():
            updates.append((LABEL_MAPPING["neutral"], 0.0, comment_id))
            continue
        
        try:
            result = sentiment_pipeline(truncated_text)[0]
            raw_label = result["label"].lower()
            score = result["score"]
            label = LABEL_MAPPING.get(raw_label, "neutral")
            updates.append((label, score, comment_id))
        except Exception as e:
            print(f"Error processing comment {comment_id}: {e}")
            updates.append(("neutral", 0.0, comment_id))
            
    return updates

# 6. Main Orchestration Function
def main():
    print("Connecting to the database...")
    try:
        conn = psycopg2.connect(db_url)
    except Exception as e:
        print(f"Database connection failed: {e}")
        return

    try:
        comments = fetch_comments_needing_sentiment(conn)
        total_comments = len(comments)
        print(f"Found {total_comments} comments requiring sentiment analysis.")

        if total_comments == 0:
            print("No processing required.")
            return

        for i in range(0, total_comments, BATCH_SIZE):
            batch = comments[i:i + BATCH_SIZE]
            print(f"Processing batch {i // BATCH_SIZE + 1} of {(total_comments + BATCH_SIZE - 1) // BATCH_SIZE} (IDs: {batch[0]['id']} to {batch[-1]['id']})...")
            
            updates = process_batch(batch)
            
            if updates:
                update_comment_sentiments(conn, updates)
                print(f"Successfully updated {len(updates)} comment sentiments in this batch.")
            else:
                print("No updates applied for this batch.")

            if i + BATCH_SIZE < total_comments:
                time.sleep(0.5)

        print("Sentiment analysis process complete.")

    finally:
        conn.close()

if __name__ == "__main__":
    main()