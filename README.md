# tech_news_fetcher

An end-to-end tech news aggregator that collects articles from dozens of sources, extracts the full content, and enriches every article with a **summary**, an extracted **topic/theme**, and the **community sentiment** gathered from the article's comments — all stored in a PostgreSQL database so you can query the tech landscape.

## The end goal

The final product is not just a feed of headlines, but an *intelligent digest* of the tech world. For every piece of news the pipeline delivers:

- **Raw article + full content** extracted from the source (APIs and RSS feeds, with full-text fallback extraction when a source only exposes a teaser).
- **A concise summary** of what the article is actually about — the source-provided summary when the API/feed offers one, otherwise generated with the SGR (semantic graph reduction) summarizer.
- **Key topics** extracted from the cleaned content with TextRank, so news can be grouped and browsed by subject.
- **A sentiment signal from the comments**: each community comment (Dev.to, Hacker News, Reddit) is classified as *positive / neutral / negative* with a confidence score by an SVM model. Aggregated per article into sentiment percentages (e.g. 60% positive, 30% neutral, 10% negative), this gives you **beforehand information about the topic** — how the community reacts to a story *before* you read it — letting you spot controversial releases, well-received papers, or brewing security panics at a glance.

Everything converges into a single PostgreSQL database (`sources`, `items`, `comments`, `vulnerabilities`, `hn_seen_ids`) that can be queried to answer questions like: *what topics are trending, what is the community mood on framework X, which new CVEs should I worry about.*

## How we get there — the steps before the final product

The project is built step by step; each step feeds the next:

### 1. Content extraction using APIs
First we aggregate raw news from a broad set of sources — REST APIs (Dev.to, ArXiv, Hacker News, GitHub, NVD) and RSS feeds (Reddit, InfoQ, DZone, Microsoft .NET Blog, VentureBeat, security blogs, …). See [Sources](#sources).

- `src/fetching_data/api_fetchers.py` — pulls every API source, decodes JSON/XML, and for Dev.to/HN additionally pulls the **comments** attached to each article.
- `src/fetching_data/rss_fetcher.py` — pulls the RSS feeds.
- `src/fetching_data/article_content_fetcher.py` — **content extraction**: when a source only provides a summary or HTML teaser, this fetches the article URL and extracts the full readable text with [trafilatura](https://trafilatura.readthedocs.io/), falling back to a headless Playwright browser for JS-heavy pages.

### 2. Deduplication and storage
- `src/cleaning/process_and_store.py` — the main processing stage. Before any enrichment it queries the database for existing article URLs and **skips articles that are already stored**, then for every new article it: cleans the raw text, reuses the source-provided summary (or generates one), extracts topics, classifies comments, and inserts everything.
- `src/cleaning/general_cleaning.py` — lower-level cleaning helpers (HTML/Markdown stripping, emoji removal, date normalization) and the NVD CVE parser.
- `src/db/loader.py` + `src/db/database.py` — loads the processed records into Postgres, initializes the schema, and tracks per-source fetch status.

### 3. Summarization (SGR)
- `src/functionalities/summary/SGR/semantic_graph_reduction.py` — the main path: builds a rich semantic graph from the cleaned article text (nouns, verbs, named entities with WordNet popularity scores), reduces it using WordNet relations (hypernyms/entailments), and generates a consolidated abstractive summary. If the source API/feed already provided a summary (e.g. Dev.to `description` or RSS `summary`), that summary is used instead and SGR is skipped.
- Alternative, offline summarizers are also explored under `src/functionalities/summary/`:
  - `BI-GRAM/` — n-gram / frequency-based extractive summarizer.
  - `GA/` — genetic-algorithm summarizer.
  - `llm_summary/` — LLM (Gemini) based summarization, previously the main path.

### 4. Topic extraction (TextRank)
- `src/functionalities/theme_reconginition/text_rank/script.py` — the main path: `extract_top_topics` tokenizes the cleaned text, lemmatizes with WordNet POS tags, builds a co-occurrence graph, runs PageRank until convergence, and returns the top ranked keyword/phrases for the article.
- Alternatives under `src/functionalities/theme_reconginition/`:
  - `llm/` — Gemini based theme labelling, previously the main path.
  - `lda/` — Latent Dirichlet Allocation topic modeling.
  - `cosin_similarity/` — embeddings + cosine similarity against known categories.

### 5. Sentiment analysis on the comments (SVM)
- `src/functionalities/sentiment_analysis/svm.py` — the main path: trains a `LinearSVC` over a TF-IDF representation of a labeled comment corpus, then classifies each article comment as positive/neutral/negative with a confidence score. The per-comment labels are aggregated per article into sentiment percentages (e.g. 60% positive, 30% neutral, 10% negative) stored on the article.
- Alternative approaches are explored under `src/functionalities/sentiment_analysis/`: `lexicon_method.py` and `analysis.py` (RoBERTa, previously the main path).

The whole flow is orchestrated by `main.py`, which runs the pipeline stages in sequence and logs everything to `data/pipeline.log`. Because each stage is incremental (deduplicated by URL and `ON CONFLICT DO NOTHING`), the pipeline can be re-run safely and it will just pick up where it left off.

## Sources

**APIs:**
- [Dev.to](https://dev.to) — `GET /api/articles` (tags: .NET, C#, Java, Spring Boot, Android, Flutter, Dart, AI, ML, Data Science, Networking, IoT, Security, Cryptography, SSH, Protocols) + article detail, full content, and comments
- [ArXiv](http://export.arxiv.org) — `GET /api/query` (categories: cs.AI, cs.LG, cs.NI)
- [Hacker News](https://hacker-news.firebaseio.com) — `GET /v0/newstories.json` + item detail `/v0/item/{id}.json` + nested comments
- [GitHub](https://api.github.com) — `GET /search/repositories` (dotnet, csharp, jee, security, cryptography, encryption, ssh) enriched with README, languages, contributor count
- [NVD](https://services.nvd.nist.gov) — `GET /rest/json/cves/2.0` (keywords: SSH, Cryptography, Kubernetes)

**RSS Feeds:**
- [Reddit](https://www.reddit.com) — `/r/dotnet/.rss`, `/r/flutterdev/.rss`, `/r/netsec/.rss`
- [InfoQ](https://feed.infoq.com) — `/dotnet/`, `/ai-ml-data-eng/`, `/security/`, `/java/`, `/development/`, `/mobile/`, `/architecture-design/`, `/culture-methods/`
- [DZone](https://feeds.dzone.com) — `/java`
- [Microsoft .NET Blog](https://devblogs.microsoft.com) — `/dotnet/feed/`
- [VentureBeat](https://venturebeat.com) — `/category/ai/feed/`
- [Light Reading](https://www.lightreading.com) — `/rss.xml`
- [The Hacker News](https://feeds.feedburner.com) — `/TheHackersNews`
- [Dark Reading](https://www.darkreading.com) — `/rss.xml`
- [Schneier on Security](https://www.schneier.com) — `/feed/atom/`
- [Packet Storm](https://rss.packetstormsecurity.com) — `/`

Sources are declared declaratively in `src/config/sources.yaml` — adding a new source is just a new entry there.

## Project structure

```
main.py                                   # Pipeline orchestrator (fetch + process)
pipeline.py                               # In-process variant of the pipeline
Makefile                                  # make pipeline / server / profile ...
src/
  fetching_data/
    api_fetchers.py                       # API + comments fetching
    rss_fetcher.py                        # RSS fetching
    article_content_fetcher.py            # Full-text content extraction
    general_api_fetcher.py                # HTTP client used by the fetchers
  cleaning/
    process_and_store.py                  # dedup → clean → SGR → TextRank → SVM → insert
    general_cleaning.py                   # Cleaning helpers + NVD parsing
  db/
    database.py                           # Connection pool + init + migrations
    loader.py                             # Inserts into Postgres
    schema.sql                            # DB schema
  functionalities/
    summary/                              # SGR (main) + BI-GRAM/GA/LLM summarizers
    theme_reconginition/                  # TextRank (main) + LLM/LDA/cosine themes
    sentiment_analysis/                   # SVM (main) + lexicon/RoBERTa sentiment
requirements.txt                          # Pinned dependencies
```

## Manual — how to use it so far

### Prerequisites

- Python 3.12
- A running PostgreSQL database
- Optional API keys: a free [NVD](https://nvd.nist.gov/developers/request-an-api-key) key, [Google Gemini](https://ai.google.dev/) keys, and optionally a GitHub token for higher rate limits.

### 1. Setup

```bash
# create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# install the Playwright browser (needed for JS-heavy article extraction)
playwright install chromium
```

### 2. Configure environment

Create a `.env` file at the project root:

```
DATABASE_URL=postgresql://user:password@localhost:5432/tech_news
NVD_API_KEY=your_nvd_key_here
GEMINI_API_KEY=your_gemini_key_here
GEMINI_SUMMARY_KEY=your_primary_summary_key_here
GEMINI_REPLACEMENT_KEY=your_secondary_summary_key_here   # optional fallback
GITHUB_TOKEN=your_github_token_here                      # optional
```

`DATABASE_URL` and a Gemini key are required for the summarization/theme steps; the rest are optional.

### 3. Run the pipeline

```bash
# full pipeline (fetch → process: dedup → clean → SGR summary → TextRank topics → SVM sentiment → insert)
make pipeline
# or
.venv/bin/python3 main.py
```

This runs:
1. `api_fetchers.py` — fetch API sources + comments
2. `rss_fetcher.py` — fetch RSS sources
3. `article_content_fetcher.py` — extract full article text
4. `process_and_store.py` — deduplicate by URL, clean, summarize (SGR), extract topics (TextRank), analyze comment sentiment (SVM), and insert everything

The schema is created automatically on the first run. Progress is logged to `data/pipeline.log`, and raw fetches land in `data/YYYY-MM-DD/*.json`. Each stage is incremental: re-running only processes articles whose URL is not already in the database.
