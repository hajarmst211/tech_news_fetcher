PYTHON = python3

FETCH_DIR = src/fetching_data
CLEAN_DIR = src/cleaning

.PHONY: fetch articles clean stats all profile visualizeProfile

all: fetch articles clean stats

fetch:
	$(PYTHON) $(FETCH_DIR)/api_fetchers.py
	$(PYTHON) $(FETCH_DIR)/rss_fetcher.py

articles:
	$(PYTHON) $(FETCH_DIR)/article_content_fetcher.py

clean:
	$(PYTHON) $(CLEAN_DIR)/general_cleaning.py

stats:
	$(PYTHON) cleaning_stats.py

main:
	$(PYTHON) main.py

profile:
	$(PYTHON) -m cProfile -s tottime -o program.prof main.py

visualizeProfile:
	snakeviz program.prof



