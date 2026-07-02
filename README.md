# tech_news_fetcher

Aggregates tech news from the following sources:

**APIs:**
- [Dev.to](https://dev.to) — `GET /api/articles` (tags: .NET, C#, Java, Spring Boot, Android, Flutter, Dart, AI, ML, Data Science, Networking, IoT, Security, Cryptography, SSH, Protocols)
- [ArXiv](http://export.arxiv.org) — `GET /api/query` (categories: cs.AI, cs.LG, cs.NI)
- [Hacker News](https://hacker-news.firebaseio.com) — `GET /v0/newstories.json` + item detail `/v0/item/{id}.json`
- [GitHub](https://api.github.com) — `GET /repos/{owner}/{repo}/releases/latest` (flutter/flutter, dotnet/runtime, PowerShell/PowerShell)
- [NVD](https://services.nvd.nist.gov) — `GET /rest/json/cves/2.0` (keywords: SSH, Cryptography, Kubernetes)

**RSS Feeds:**
- [Reddit](https://www.reddit.com) — `/r/dotnet/.rss`, `/r/flutterdev/.rss`, `/r/netsec/.rss`
- [InfoQ](https://feed.infoq.com) — `/dotnet/`
- [DZone](https://feeds.dzone.com) — `/java`
- [Microsoft .NET Blog](https://devblogs.microsoft.com) — `/dotnet/feed/`
- [VentureBeat](https://venturebeat.com) — `/category/ai/feed/`
- [Light Reading](https://www.lightreading.com) — `/rss.xml`
- [The Hacker News](https://feeds.feedburner.com) — `/TheHackersNews`
- [Dark Reading](https://www.darkreading.com) — `/rss.xml`
- [Schneier on Security](https://www.schneier.com) — `/feed/atom/`
- [Packet Storm](https://rss.packetstormsecurity.com) — `/`



Primary Bottleneck: External subprocess execution. Over 99.5% of the program's runtime is spent waiting for external commands to finish.


Content (per source type):
- Dev.to API — articles with title, description, url, tags, user, published_at, etc.
- ArXiv API (XML) — entries with id, title, summary, authors, categories, published
- Hacker News API — story items with title, url, score, by, time, descendants, optional text
- GitHub Releases API — release objects with tag_name, name, body (Markdown), published_at
- NVD API — vulnerabilities[] with cve.id, descriptions, metrics.cvssMetricV2, weaknesses
- RSS feeds (Reddit, InfoQ, DZone, VentureBeat, etc.) — entries with title, link, published, summary (HTML), content (HTML), tags