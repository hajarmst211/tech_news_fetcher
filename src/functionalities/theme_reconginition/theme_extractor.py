import json
import os
import time
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import DictCursor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
env_path = PROJECT_ROOT / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

api_key_primary = os.environ.get("GEMINI_SUMMARY_KEY") or os.environ.get("GEMINI_API_KEY")
api_key_replacement = os.environ.get("GEMINI_REPLACEMENT_KEY")

if not api_key_primary:
    raise ValueError("Please set the GEMINI_SUMMARY_KEY or GEMINI_API_KEY environment variable.")


db_url = os.environ.get("DATABASE_URL")
if not db_url:
    raise ValueError("Please set the DATABASE_URL environment variable.")

current_key = api_key_primary
client = genai.Client(api_key=current_key)

BATCH_SIZE = 15  
MODEL_NAME = "gemini-2.5-flash"  

def load_prompt_template() -> str:
    prompt_path = Path(__file__).parent / "prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Could not find prompt.txt at {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

def fetch_articles_needing_theme(conn) -> list:
    query = """
        SELECT id, summary AS content 
        FROM items 
        WHERE summary IS NOT NULL 
          AND TRIM(summary) != ''
          AND (theme IS NULL OR TRIM(theme) = '')
        ORDER BY id ASC;
    """
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(query)
        return [dict(row) for row in cur.fetchall()]

def update_article_themes(conn, updates: list):
    query = """
        UPDATE items 
        SET theme = %s 
        WHERE id = %s;
    """
    with conn.cursor() as cur:
        for update in updates:
            cur.execute(query, (update["theme"], update["id"]))
    conn.commit()

def process_batch(batch: list, prompt_template: str) -> list:
    global client, current_key
    formatted_input = [{"id": item["id"], "content": item["content"]} for item in batch]
    formatted_prompt = prompt_template.replace("{articles_json}", json.dumps(formatted_input, indent=2))
    
    max_retries = 3
    retry_delay = 35.0

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=formatted_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2, 
                ),
            )
            
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text.split("```json", 1)[1]
            if response_text.endswith("```"):
                response_text = response_text.rsplit("```", 1)[0]
            
            themes = json.loads(response_text.strip())
            return themes if isinstance(themes, list) else []

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                print(f"Rate limit hit on attempt {attempt + 1} of {max_retries}.")
                if current_key == api_key_primary and api_key_replacement:
                    print("Switching to GEMINI_REPLACEMENT_KEY...")
                    current_key = api_key_replacement
                    client = genai.Client(api_key=current_key)
                    continue
                
                if attempt < max_retries - 1:
                    print(f"Waiting {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
                else:
                    print("Max retries exceeded for this batch.")
            else:
                print(f"API Error processing batch: {e}")
                break

    return []

def main():
    print("Connecting to the database...")
    try:
        conn = psycopg2.connect(db_url)
    except Exception as e:
        print(f"Database connection failed: {e}")
        return

    try:
        prompt_template = load_prompt_template()
        articles = fetch_articles_needing_theme(conn)
        total_articles = len(articles)
        print(f"Found {total_articles} articles requiring theme extraction.")

        if total_articles == 0:
            print("No processing required.")
            return

        for i in range(0, total_articles, BATCH_SIZE):
            batch = articles[i:i + BATCH_SIZE]
            print(f"Processing batch {i // BATCH_SIZE + 1} of {(total_articles + BATCH_SIZE - 1) // BATCH_SIZE} (IDs: {batch[0]['id']} to {batch[-1]['id']})...")
            
            themes = process_batch(batch, prompt_template)
            
            if themes:
                valid_updates = []
                batch_ids = {item["id"] for item in batch}
                
                for t in themes:
                    try:
                        item_id = int(t.get("id"))
                        if item_id in batch_ids and t.get("theme"):
                            valid_updates.append({"id": item_id, "theme": t["theme"]})
                    except (ValueError, TypeError):
                        continue

                if valid_updates:
                    update_article_themes(conn, valid_updates)
                    print(f"Successfully updated {len(valid_updates)} themes in this batch.")
                else:
                    print("No valid updates found in API response for this batch.")
            else:
                print("Skipping batch due to processing errors.")

            if i + BATCH_SIZE < total_articles:
                time.sleep(5)

        print("Theme extraction process complete.")

    finally:
        conn.close()

if __name__ == "__main__":
    main()