# Database Schema

## Overview

PostgreSQL database for storing fetched tech news from various sources (Dev.to, ArXiv, GitHub, NVD, RSS feeds, Hacker News).

## Tables

### `sources`

One row per feed/API endpoint pulled from.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | `SMALLSERIAL` | `PRIMARY KEY` | Auto-incrementing ID |
| `name` | `TEXT` | `NOT NULL UNIQUE` | Sanitized source name (e.g. `devto_ai`, `arxiv_networks`) |
| `source_type` | `TEXT` | `NOT NULL` | `devto`, `arxiv`, `github_release`, `nvd`, `rss`, `hn` |
| `category` | `TEXT` | | Optional category (e.g. `ai`, `security`, `flutter`) |

### `items`

Core table covering Dev.to articles, ArXiv papers, GitHub releases, Reddit posts, RSS blog entries, and HN item details.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | `BIGSERIAL` | `PRIMARY KEY` | Auto-incrementing ID |
| `source_id` | `SMALLINT` | `NOT NULL REFERENCES sources(id)` | FK to sources table |
| `external_id` | `TEXT` | `NOT NULL` | Source-specific ID (devto id, arxiv id, tag_name, HN id, RSS guid) |
| `title` | `TEXT` | `NOT NULL` | Article/release/post title |
| `summary` | `TEXT` | | Description or short summary |
| `url` | `TEXT` | `NOT NULL` | Link to the original |
| `author` | `TEXT` | | Flattened author name/username |
| `published_at` | `TIMESTAMPTZ` | | Original publication timestamp |
| `updated_at` | `TIMESTAMPTZ` | | Last update timestamp |
| `fetched_at` | `TIMESTAMPTZ` | `NOT NULL DEFAULT now()` | When we fetched it |
| `content` | `TEXT` | | Full cleaned plain-text content |
| `content_hash` | `CHAR(64)` | | SHA-256 of content, for cross-source dedup |
| `tags` | `TEXT[]` | | Normalized tag array (one canonical form) |
| `metrics` | `JSONB` | | Small frequently-queried numbers: comments_count, score, stars, reactions |
| `extra` | `JSONB` | | Everything else source-specific that's rarely queried |

**Indexes:**
- `idx_items_published` — `(published_at DESC)` for feed-style queries
- `idx_items_hash` — `(content_hash)` for dedup lookups
- `idx_items_tags` — GIN index on `tags` for tag filtering

**Unique constraint:** `(source_id, external_id)` — each source's external IDs are unique.

### `vulnerabilities`

NVD data is structurally different from articles — kept separate.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `cve_id` | `TEXT` | `PRIMARY KEY` | CVE identifier (e.g. `CVE-2025-12345`) |
| `source_id` | `SMALLINT` | `NOT NULL REFERENCES sources(id)` | FK to sources table |
| `description` | `TEXT` | | English description of the vulnerability |
| `published_at` | `TIMESTAMPTZ` | | CVE publication date |
| `last_modified` | `TIMESTAMPTZ` | | Last modification date |
| `cvss_score` | `NUMERIC(3,1)` | | CVSS base score (0.0 – 10.0) |
| `cvss_severity` | `TEXT` | | Severity label (NONE, LOW, MEDIUM, HIGH, CRITICAL) |
| `cvss_vector` | `TEXT` | | CVSS vector string |
| `raw` | `JSONB` | `NOT NULL` | Full original CVE object from the NVD API |

### `hn_seen_ids`

The Hacker News "new stories" endpoint returns an ordered list of story IDs. This table tracks which IDs we've already seen, acting as a dedup queue.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `hn_id` | `BIGINT` | `PRIMARY KEY` | Hacker News story ID |
| `seen_at` | `TIMESTAMPTZ` | `NOT NULL DEFAULT now()` | When we first observed this ID |

## Data Flow

```
raw JSON (data/raw/YYYY-MM-DD/)
    │
    ▼
clean in memory (general_cleaning.py)
    │  clean_text() — strip HTML + markdown
    │  extract_nvd_fields() — source-specific NVD extraction
    │  classify_source() → dispatch to right handler
    │  dedup_records() — batch dedup by external_id
    ▼
insert into DB (db/loader.py)
    │  insert_items()       → items table
    │  insert_vulnerabilities() → vulnerabilities table
    │  insert_hn_seen_ids() → hn_seen_ids table
    ▼
PostgreSQL
```

No intermediate cleaned JSON files are written — data goes straight from raw JSON through in-memory cleaning to the database.
