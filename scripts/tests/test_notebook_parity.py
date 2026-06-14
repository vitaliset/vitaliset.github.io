"""Parity tests: the EN and PT notebooks of a bilingual post must stay parallel.

A bilingual post (see DEVELOPMENT.md) shares ONE set of code/figures across both
languages and pairs each prose paragraph EN<->PT. That only holds if the two
notebooks carry the same content in the same order. These tests lock that
invariant so the pair can't silently drift: edit one notebook without the other
and they fail loudly.

Comparison is at the *paragraph/token* level, NOT cell-by-cell: a markdown cell
may legitimately be split differently between the two notebooks (e.g. EN keeps a
section in one cell, PT splits it in two). What must match is the flattened stream
of code blocks, headings, prose paragraphs, rules and images.

Complements test_prose_sync.py (each post track <-> its own notebook); here we
compare the two notebooks directly.
"""
import re
from pathlib import Path

import pytest

import notebook_to_post as n

REPO_ROOT = Path(__file__).resolve().parents[2]

# (id, EN notebook, PT notebook) for every bilingual notebook pair.
PARITY_PAIRS = [
    pytest.param(
        "code/covariate_introduction/covariate_shift_introduction_EN.ipynb",
        "code/covariate_introduction/covariate_shift_introduction_PTBR.ipynb",
        id="covariate_introduction",
    ),
    pytest.param(
        "code/r_squared/blog_r2_score_EN.ipynb",
        "code/r_squared/blog_r2_score_PT.ipynb",
        id="r_squared",
    ),
    pytest.param(
        "code/distance_metrics/blog_distance_metrics_EN.ipynb",
        "code/distance_metrics/blog_distance_metrics_PT.ipynb",
        id="distance_metrics",
    ),
]


def _load(rel):
    import nbformat

    return nbformat.read(str(REPO_ROOT / rel), as_version=4)


def _tokens(nb):
    """Flatten cells to a token stream, ignoring markdown-cell boundaries.

    Each code cell is one ('code', directive-stripped source) token; each markdown
    cell is split on blank lines into ('head'|'rule'|'img'|'prose', text) tokens.
    """
    toks = []
    for c in nb.cells:
        if c.cell_type == "code":
            toks.append(("code", n._cell_directive(c.source)[1].strip()))
        elif c.cell_type == "markdown":
            for block in c.source.split("\n\n"):
                s = block.strip()
                if not s:
                    continue
                if re.fullmatch(r"[-_*]{3,}", s):
                    typ = "rule"
                elif s.startswith("#"):
                    typ = "head"
                elif "![" in s or "<img" in s:
                    typ = "img"
                else:
                    typ = "prose"
                toks.append((typ, s))
    return toks


def _img_count(cell):
    """Number of image outputs a code cell produces (-> output_<cell>_<n> figures)."""
    return sum(
        1
        for out in cell.get("outputs", [])
        if any(k.startswith("image/") for k in out.get("data", {}))
    )


@pytest.mark.parametrize("en_rel,pt_rel", PARITY_PAIRS)
def test_notebooks_are_parallel(en_rel, pt_rel):
    en, pt = _load(en_rel), _load(pt_rel)
    te, tp = _tokens(en), _tokens(pt)

    # Same flattened structure: markdown<->markdown / code<->code line up, and
    # every prose paragraph and heading has exactly one twin to pair in the post.
    assert [t for t, _ in te] == [t for t, _ in tp], "token-type sequence differs (EN vs PT)"

    # Code is shared in the post -> must be byte-identical (directive line aside).
    code_en = [s for t, s in te if t == "code"]
    code_pt = [s for t, s in tp if t == "code"]
    assert code_en == code_pt, "a code block differs between EN and PT"

    # Figures come from the EN run; each code cell must emit the same image count.
    imgs_en = [_img_count(c) for c in en.cells if c.cell_type == "code"]
    imgs_pt = [_img_count(c) for c in pt.cells if c.cell_type == "code"]
    assert imgs_en == imgs_pt, "image-output counts differ between EN and PT"
