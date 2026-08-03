import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

LIMIT = 100
OUTPUT_FILE = "comments.md"

QUERY = """
WITH ranked AS (
    SELECT c.id, s.name AS source, c.author, c.body_text, c.published_at,
           ROW_NUMBER() OVER (PARTITION BY s.id ORDER BY random()) AS rn
    FROM comments c
    JOIN items i ON i.id = c.item_id
    JOIN sources s ON s.id = i.source_id
)
SELECT source, author, body_text, published_at
FROM ranked
ORDER BY rn
LIMIT %s
"""

# Fetch data from the database
conn = psycopg2.connect(DATABASE_URL)
with conn.cursor() as cur:
    cur.execute(QUERY, (LIMIT,))
    rows = cur.fetchall()
conn.close()

# Write the output to a Markdown file
try:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("# Exported Comments\n\n")
        
        for source, author, body, published_at in rows:
            # Format header for each comment
            f.write(f"### [{source}] {author}\n")
            if published_at:
                f.write(f"*Published at: {published_at}*\n\n")
            else:
                f.write("\n")
            
            # Format body text as a blockquote
            # Replacing inner newlines with blockquote markers to maintain markdown structure
            formatted_body = body.replace("\n", "\n> ")
            f.write(f"> {formatted_body}\n\n")
            
            # Horizontal rule separator
            f.write("---\n\n")
            
    print(f"Successfully wrote {len(rows)} comments to {OUTPUT_FILE}")

except IOError as e:
    print(f"An error occurred while writing to the file: {e}")