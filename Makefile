# Developer shortcuts. Tutorials are marimo notebooks (notebooks/*.py); see notebooks/README.md.
PYTHON ?= python
NOTEBOOKS := $(sort $(wildcard notebooks/[0-9][0-9]_*.py))

.PHONY: docs docs-build notebooks-export notebooks-check notebooks-test assets

# Export every notebook to docs/notebooks/*.ipynb (with outputs), post-process, then serve the site.
docs: notebooks-export
	mkdocs serve

# Same as CI: strict build into site/
docs-build: notebooks-export
	mkdocs build --strict

notebooks-export:
	mkdir -p docs/notebooks
	@for nb in $(NOTEBOOKS); do \
	  echo "exporting $$nb"; \
	  TGMM_DOCS_FAST=1 MPLBACKEND=Agg marimo export ipynb "$$nb" \
	    -o "docs/notebooks/$$(basename $${nb%.py}).ipynb" --include-outputs --sort top-down || exit 1; \
	done
	$(PYTHON) scripts/postprocess_notebooks.py docs/notebooks

notebooks-check:
	marimo check --ignore-scripts notebooks/

notebooks-test:
	$(PYTHON) -m pytest -m notebooks tests/test_notebooks.py -v

# Re-render the videos the tutorials load (notebooks/assets/*.mp4). Slow (minutes): the notebooks only show the finished files.
assets:
	$(PYTHON) scripts/make_assets/make_em_videos.py
	$(PYTHON) scripts/make_assets/make_seeding_videos.py
