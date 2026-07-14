PYTHON = .venv/bin/python3 

FETCH_DIR = src/fetching_data
CLEAN_DIR = src/cleaning

.PHONY: fetch pipeline articles clean stats all profile visualizeProfile

pipeline:
	$(PYTHON) main.py

stats:
	$(PYTHON) generate_stats.py

all: pipeline stats

server:
	$(PYTHON) -m http.server 8000

profile:
	$(PYTHON) -m cProfile -s tottime -o program.prof main.py

visualizeProfile:
	snakeviz program.prof



