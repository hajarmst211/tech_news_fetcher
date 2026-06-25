PYTHON = python3

FETCH_DIR = src/fetching_data
CLEAN_DIR = src/cleaning

.PHONY: fetch clean all profile snakeviz

all: main

fetch:
	$(PYTHON) $(FETCH_DIR)/api_fetchers.py
	$(PYTHON) $(FETCH_DIR)/rss_fetcher.py

clean:
	$(PYTHON) $(CLEAN_DIR)/general_cleaning.py

main:
	$(PYTHON) main.py

profile:
	$(PYTHON) -m cProfile -s tottime main.py

snakeviz:
	$(PYTHON) -m cProfile -o program.prof main.py
	snakeviz program.prof
