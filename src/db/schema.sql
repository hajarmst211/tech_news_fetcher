CREATE TABLE IF NOT EXISTS sources (
    id                SMALLSERIAL PRIMARY KEY,
    name              TEXT NOT NULL UNIQUE,
    source_type       TEXT NOT NULL,
    category          TEXT,
    last_fetch_status TEXT,
    last_status_code INTEGER,
    last_fetched_at   TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS items (
    id              BIGSERIAL PRIMARY KEY,
    source_id       SMALLINT NOT NULL REFERENCES sources(id),
    external_id     TEXT NOT NULL,
    title           TEXT NOT NULL,
    summary         TEXT,
    theme           TEXT,
    url             TEXT NOT NULL,
    author          TEXT,
    published_at    TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    content         TEXT,
    content_hash    CHAR(64),
    tags            TEXT[],
    metrics         JSONB,
    extra           JSONB,
    UNIQUE (source_id, external_id)
);

CREATE INDEX IF NOT EXISTS idx_items_published ON items (published_at DESC);
CREATE INDEX IF NOT EXISTS idx_items_hash ON items (content_hash);
CREATE INDEX IF NOT EXISTS idx_items_tags ON items USING GIN (tags);

DO $$ BEGIN
    CREATE TYPE sentiment_label_enum AS ENUM ('positive', 'neutral', 'negative');
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

CREATE TABLE IF NOT EXISTS comments (
    id              BIGSERIAL PRIMARY KEY,
    item_id         BIGINT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    external_id     TEXT NOT NULL,
    author          TEXT,
    body_text       TEXT,
    published_at    TIMESTAMPTZ,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    extra                 JSONB,
    sentiment_label       sentiment_label_enum,
    sentiment_score       REAL,
    UNIQUE (item_id, external_id)
);

CREATE INDEX IF NOT EXISTS idx_comments_item ON comments (item_id);
CREATE INDEX IF NOT EXISTS idx_comments_published ON comments (published_at DESC);
CREATE INDEX IF NOT EXISTS idx_comments_sentiment ON comments (sentiment_label);

CREATE TABLE IF NOT EXISTS vulnerabilities (
    cve_id          TEXT PRIMARY KEY,
    source_id       SMALLINT NOT NULL REFERENCES sources(id),
    description     TEXT,
    published_at    TIMESTAMPTZ,
    last_modified   TIMESTAMPTZ,
    cvss_score      NUMERIC(3,1),
    cvss_severity   TEXT,
    cvss_vector     TEXT,
    raw             JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS hn_seen_ids (
    hn_id   BIGINT PRIMARY KEY,
    seen_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
---item_data = api.get_item_details(item_id)