# scripts/Makefile — notebook -> post tooling (uv-based)
#
# Run from anywhere via -C:
#   make -C scripts post NOTEBOOK=code/<slug>/file.ipynb SLUG=<slug> DATE=YYYY-MM-DD \
#        [TITLE="..."] [CATEGORY="🇺🇸, basic"] [DESCRIPTION="..."]
#   make -C scripts test
#
# uv runs in *script mode* (--no-project): it uses the PEP 723 dependencies declared
# in notebook_to_post.py and a global cache, so it never adopts or modifies the
# per-notebook projects under code/ (Poetry today, uv later) and writes no repo
# artifacts. Paths below are relative to the repo root.

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
UV   := uv run --no-project

# Optional front-matter overrides (passed through only when set).
TITLE       ?=
CATEGORY    ?=
DESCRIPTION ?=

.PHONY: help post test

help:
	@echo "make -C scripts post NOTEBOOK=code/<slug>/file.ipynb SLUG=<slug> DATE=YYYY-MM-DD [TITLE=..] [CATEGORY=..] [DESCRIPTION=..]"
	@echo "make -C scripts test"

post:
	@test -n "$(NOTEBOOK)" || { echo "NOTEBOOK is required"; exit 1; }
	@test -n "$(SLUG)"     || { echo "SLUG is required"; exit 1; }
	@test -n "$(DATE)"     || { echo "DATE is required (YYYY-MM-DD)"; exit 1; }
	@cd "$(ROOT)" && $(UV) scripts/notebook_to_post.py \
		--notebook "$(NOTEBOOK)" --slug "$(SLUG)" --date "$(DATE)" \
		$(if $(TITLE),--title "$(TITLE)",) \
		$(if $(CATEGORY),--category "$(CATEGORY)",) \
		$(if $(DESCRIPTION),--description "$(DESCRIPTION)",)

test:
	@cd "$(ROOT)" && $(UV) --with pytest --with nbconvert --with nbformat pytest scripts/tests/
