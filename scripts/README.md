# scripts/ — notebook → post tooling

Tooling for turning a Jupyter notebook in [`code/<slug>/`](../code) into a Jekyll
post scaffold under [`_posts/`](../_posts), with figures extracted to
`assets/img/<slug>/`. Excluded from the Jekyll build via `_config.yml`.

## What `notebook_to_post.py` does

It is a **scaffolding** tool, not a full publisher. Running it produces:

- code cells as ` ```python ` fenced blocks (no indentation),
- cell stdout / text outputs as 4-space-indented blocks,
- figures saved as `assets/img/<slug>/output_<cell>_<out>.png` and embedded as
  `<p><div align="justify"><center><img src="{{ site.baseurl }}/assets/img/<slug>/output_X_Y.png"></center></div></p>`,
- front matter + an "experiments" link pointing at `code/<slug>`.

Prose from the notebook's markdown cells comes through as plain markdown. The
published posts wrap prose in justified HTML (`<p><div align="justify">…</div></p>`),
which nbconvert can't reproduce, so **prose is hand-authored afterward**. The
script removes the mechanical steps and stays byte-consistent with the recent
posts for the code/output/image blocks (enforced by `tests/`).

## Usage

Recommended (zero-setup, via [uv](https://docs.astral.sh/uv/) — reads the inline
PEP 723 dependencies in the script header):

```sh
uv run scripts/notebook_to_post.py \
    --notebook code/<slug>/<file>.ipynb --slug <slug> --date YYYY-MM-DD \
    --title "Post title" --category "🇺🇸, basic" --summary "One-liner."
```

Via the Makefile (wraps the uv command, kept inside `scripts/`):

```sh
make -C scripts post NOTEBOOK=code/<slug>/file.ipynb SLUG=<slug> DATE=YYYY-MM-DD \
     TITLE="Post title" CATEGORY="🇺🇸, basic" SUMMARY="One-liner."
make -C scripts test
```

Plain Python fallback:

```sh
pip install -r scripts/requirements.txt
python scripts/notebook_to_post.py --notebook code/<slug>/<file>.ipynb --slug <slug> --date YYYY-MM-DD
```

After running, hand-author the prose (wrap it in justified HTML) before publishing.

### uv isolation (no conflict with `code/` projects)

`uv run --no-project` runs the tool in *script mode*: it resolves the PEP 723
dependencies declared at the top of `notebook_to_post.py` into uv's global cache
and never adopts a surrounding project or writes a `.venv`/lockfile into the repo.
So it does **not** interfere with the per-notebook environments under `code/`
— whether those use Poetry (today) or uv (later). When a `code/<slug>/` project
is migrated to uv, run that notebook with `uv run` *from inside its own folder*;
that is independent of this tool and needs no changes here. (Avoid placing a
`pyproject.toml` at the repo root, which would make uv treat `code/<slug>/` as
workspace members.)

## Directives

The notebook stays the single source of truth. To reconcile a notebook's cells
with how a post should render, put a directive comment as the **first line** of a
code cell. The directive line is stripped from the rendered post (it stays in the
notebook and is a harmless comment when the notebook runs):

| Directive | Effect |
| --- | --- |
| `# nb2post: skip` | Drop this cell from the post (code + output). |
| `# nb2post: skip-input` | Hide the code but keep its output (e.g. show only a figure or table, not the plotting boilerplate). |
| `# nb2post: skip-output` | Keep the code, drop its output. |
| `# nb2post: merge` | Append this cell to the previous code block (one ` ```python ` fence). |

Consumed cells are replaced by an empty placeholder rather than deleted (and
`skip-input` keeps its cell in place, just tagged for nbconvert), so the absolute
cell index — and therefore the `output_<cell>_<out>.png` figure numbering — stays
stable.

DataFrame outputs are left as nbconvert's HTML table; if an older post rendered
such a table as an image, update the post to the HTML and drop the image (the
notebook is the source of truth).

## Tests

```sh
uv run --with pytest --with nbconvert --with nbformat pytest scripts/tests/
# or, with deps installed:
python -m pytest scripts/tests/
```

The snapshot tests run the pipeline on real notebooks and compare the generated
code/output/image blocks to the committed posts byte-for-byte (prose ignored).

**Strictly tested posts:** `r_squared`, `evaluating_ranking_in_regression`,
`metakmeans`, `covariate_introduction`, `cqr_cate`, and `threshold_dependent_opt`.
These regenerate exactly from their notebooks — `evaluating_ranking` via
`nb2post:merge`; the others via `skip`/`skip-input`/`skip-output` plus reconciling
their DataFrame tables to HTML (and normalizing older plain embeds to div-align).

**Future work — strict for all posts.** The remaining notebook-backed posts
(`boruta`, `conditional_density_estimation`) are not byte-reproducible yet:
`conditional_density_estimation` has genuine non-table figures, and `boruta` is
mostly table-as-image + suppressed outputs but also has a `%%time` magic and a
hand-edited printed output. Bringing each under strict tests means adding the
per-post directives above and normalizing any plain `<p><center><img>` embeds to
the `<div align="justify">` form. The goal is for every post to be strict.
