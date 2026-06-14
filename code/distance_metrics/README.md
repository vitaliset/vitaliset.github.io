# Generalizing distance

Reproducible source for the *"Generalizing distance"* post on
[vitaliset.github.io](https://vitaliset.github.io/distancia/).

The notebook builds intuition for the mathematical definition of a metric and
plots the open balls of several classic distances (Minkowski / Chebyshev, the
discrete metric, and weighted metrics), alongside examples used in the data
science context (Hamming, angular/cosine, the sup distance between functions).

## Files

| File | Description |
| ---- | ----------- |
| `blog_distance_metrics_EN.ipynb` | Main notebook, English prose (fully executed) |
| `blog_distance_metrics_PT.ipynb` | Portuguese prose; same (English) code and figures |
| `pyproject.toml` / `uv.lock` | Reproducible uv environment |

The post is **bilingual** (EN/PT toggle on a single URL): the code and figures are
shared and only the prose differs, so the two notebooks are kept parallel
(`scripts/tests/test_notebook_parity.py`). They are the single source of truth;
the Jekyll post is regenerated from them with `scripts/notebook_to_post.py`, using
`# nb2post:` directives to hide the plotting boilerplate behind the figures it shows.

## Running the environment

This project uses [uv](https://docs.astral.sh/uv/). From inside this folder:

```sh
uv sync                    # create .venv and install all dependencies
uv run jupyter notebook    # launch Jupyter; open blog_distance_metrics_EN.ipynb
```

## Linting & formatting

A minimal [ruff](https://docs.astral.sh/ruff/) setup is configured in
`pyproject.toml` (`E`/`F`/`W`/`I`/`B` rules; a few rules are relaxed to keep the
teaching code in the post faithful — see the config comments):

```sh
uv run ruff check .          # report issues
uv run ruff format .         # format the code (also formats notebook cells)
```
