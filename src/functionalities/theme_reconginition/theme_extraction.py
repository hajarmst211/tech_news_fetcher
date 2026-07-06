import os
import psycopg2
from psycopg2.extras import execute_batch
from google import genai
from pathlib import Path
from dotenv import load_dotenv  

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
db_url = os.environ.get("DATABASE_URL")

if not api_key:
    raise ValueError("Please set the GEMINI_API_KEY environment variable.")

if not db_url:
    raise ValueError("Please set the DATABASE_URL environment variable.")

client = genai.Client(api_key=api_key)

def load_prompt_template(prompt_filename="prompt.txt"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, prompt_filename)
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file '{prompt_path}' not found.")
        return None

def get_theme(article_text, prompt_template):
    if not article_text or not article_text.strip():
        return None

    full_prompt = prompt_template.format(text=article_text)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
        )
        return response.text.strip() if response.text else None
    except Exception as e:
        print(f"Error communicating with the Gemini API: {e}")
        return None

def process_themes_in_batches(batch_size=50):
    prompt_template = load_prompt_template()
    if not prompt_template:
        return

    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()

        # Ensure the theme column exists in the items table
        cursor.execute("ALTER TABLE items ADD COLUMN IF NOT EXISTS theme TEXT;")
        conn.commit()

        # Select items that have a summary but do not yet have a theme.
        # This allows the script to be resumed if interrupted.
        select_query = """
            SELECT id, summary 
            FROM items 
            WHERE summary IS NOT NULL AND (theme IS NULL OR theme = '')
            ORDER BY id;
        """
        cursor.execute(select_query)
        
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            updates = []
            print(f"Processing batch of {len(rows)} items...")

            for item_id, summary in rows:
                theme = get_theme(summary, prompt_template)
                if theme:
                    updates.append((theme, item_id))

            if updates:
                update_query = "UPDATE items SET theme = %s WHERE id = %s;"
                # execute_batch is more efficient for batch updates than executing one by one
                execute_batch(cursor, update_query, updates)
                conn.commit()
                print(f"Successfully updated {len(updates)} items in this batch.")

        cursor.close()
        conn.close()
        print("Processing complete.")

    except Exception as e:
        print(f"An error occurred during database operations: {e}")

if __name__ == "__main__":
    process_themes_in_batches(batch_size=50)