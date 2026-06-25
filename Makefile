PYTHON = python3

FETCH_DIR = src/fetching_data
CLEAN_DIR = src/cleaning

.PHONY: fetch clean stats all profile visualizeProfile

all: fetch clean stats

fetch:
	$(PYTHON) $(FETCH_DIR)/api_fetchers.py
	$(PYTHON) $(FETCH_DIR)/rss_fetcher.py

clean:
	$(PYTHON) $(CLEAN_DIR)/general_cleaning.py

stats:
	$(PYTHON) cleaning_stats.py

main:
	$(PYTHON) main.py

profile:
	$(PYTHON) -m cProfile -s tottime main.py

visualizeProfile:
	$(PYTHON) -m cProfile -o program.prof main.py
	snakeviz program.prof
