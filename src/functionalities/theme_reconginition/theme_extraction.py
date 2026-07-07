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

api_key = os.environ.get("GEMINI_API_KEY")
db_url = os.environ.get("DATABASE_URL")

if not api_key:
    raise ValueError("Please set the GEMINI_API_KEY environment variable.")

if not db_url:
    raise ValueError("Please set the DATABASE_URL environment variable.")

client = genai.Client(api_key=api_key)

BATCH_SIZE = 50  
MODEL_NAME = "gemini-2.5-flash"  

def load_prompt_template() -> str:
    """Loads the prompt template from prompt.txt relative to this script."""
    prompt_path = Path(__file__).parent / "prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Could not find prompt.txt at {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

def fetch_articles_needing_theme(conn) -> list:
    """Fetches articles that have a summary but lack a theme."""
    query = """
        SELECT id, summary 
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
    """Updates the themes in the database in a single transaction."""
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
    """Formulates the prompt, calls Gemini API, and returns parsed themes."""
    formatted_input = [{"id": item["id"], "summary": item["summary"]} for item in batch]
    formatted_prompt = prompt_template.replace("{articles_json}", json.dumps(formatted_input, indent=2))
    
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
        print(f"API Error processing batch: {e}")
        return []

def main():
    print("Connecting to the database...")
    try:
        conn = psycopg2.connect(db_url)
    except Exception as e:
        print(f"Database connection failed: {e}")
        return

    try:
        # Ensure the column exists
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE items ADD COLUMN IF NOT EXISTS theme TEXT;")
        conn.commit()

        prompt_template = load_prompt_template()
        articles = fetch_articles_needing_theme(conn)
        total_articles = len(articles)
        print(f"Found {total_articles} articles requiring theme generation.")

        if total_articles == 0:
            print("No processing required.")
            return

        for i in range(0, total_articles, BATCH_SIZE):
            batch = articles[i:i + BATCH_SIZE]
            print(f"\nProcessing batch {i // BATCH_SIZE + 1} of {(total_articles + BATCH_SIZE - 1) // BATCH_SIZE} (IDs: {batch[0]['id']} to {batch[-1]['id']})...")
            
            themes = process_batch(batch, prompt_template)
            
            if themes:
                valid_updates = []
                batch_ids = {item["id"] for item in batch}
                
                for t in themes:
                    try:
                        item_id = int(t.get("id"))
                        theme_text = t.get("theme")
                        if item_id in batch_ids and theme_text:
                            valid_updates.append({"id": item_id, "theme": theme_text})
                            print(f"ID: {item_id:5d} | Selected Theme: {theme_text}")
                    except (ValueError, TypeError):
                        continue

                if valid_updates:
                    update_article_themes(conn, valid_updates)
                    print(f"Successfully updated {len(valid_updates)} articles in this batch.")
                else:
                    print("No valid updates found in API response for this batch.")
            else:
                print("Skipping batch due to processing error.")
            
            # Optional: short sleep between batches to remain well within free tier limits
            time.sleep(2)

        print("\nTheme categorization process complete.")

    finally:
        conn.close()

if __name__ == "__main__":
    main()