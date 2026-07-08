PYTHON = .venv/bin/python3 

FETCH_DIR = src/fetching_data
CLEAN_DIR = src/cleaning

.PHONY: fetch pipeline articles clean stats all profile visualizeProfile

fetch:
	$(PYTHON) $(FETCH_DIR)/api_fetchers.py
	$(PYTHON) $(FETCH_DIR)/rss_fetcher.py
	$(PYTHON) $(CLEAN_DIR)/general_cleaning.py

pipeline: fetch
	$(PYTHON) $(FETCH_DIR)/article_content_fetcher.py
	$(PYTHON) cleaning_stats.py

all: pipeline
	$(PYTHON) main.py


profile:
	$(PYTHON) -m cProfile -s tottime -o program.prof main.py

visualizeProfile:
	snakeviz program.prof



