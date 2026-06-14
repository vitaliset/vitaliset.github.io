"""Snapshot + unit tests for the notebook -> post conversion pipeline.

The snapshot tests lock the *mechanical* output of the pipeline (code fences,
4-space-indented outputs, and the div-align image embed) to the convention used
by the recent, in-sync posts. They run the pipeline on a real notebook and
compare the generated code/output/image blocks against the committed post
byte-for-byte. Hand-authored prose is ignored.

Currently two posts are strictly reproducible and therefore tested:

* ``r_squared``                    -> 2023-10-12-r2-score.md (no directives needed)
* ``evaluating_ranking_in_regression`` -> 2024-11-17-...md  (uses nb2post:merge)

The other notebook-backed posts contain hand-added figures and heavily
suppressed outputs, so they are not byte-reproducible from their notebooks yet.
Making them strict (per-post directives + figure handling) is tracked as future
work; see scripts/README.md.
"""
from pathlib import Path

import pytest

import notebook_to_post as n

REPO_ROOT = Path(__file__).resolve().parents[2]

STRICT_FIXTURES = [
    pytest.param(
        "r_squared",
        "code/r_squared/blog_r2_score_EN.ipynb",
        "_posts/2023-10-12-r2-score.md",
        id="r_squared",
    ),
    pytest.param(
        "evaluating_ranking_in_regression",
        "code/evaluating_ranking_in_regression/blog_evaluating_ranking_in_regression.ipynb",
        "_posts/2024-11-17-evaluating-ranking-in-regression.md",
        id="evaluating_ranking_in_regression",
    ),
    pytest.param(
        "metakmeans",
        "code/metakmeans/blog_metakmeans.ipynb",
        "_posts/2022-10-23-metakmeans.md",
        id="metakmeans",
    ),
    pytest.param(
        # asset/embed slug keeps its number; the code/ folder dropped it.
        "covariate_0_introduction",
        "code/covariate_introduction/covariate_shift_introduction_EN.ipynb",
        "_posts/2020-08-02-covariate-shift-0-introduction.md",
        id="covariate_introduction",
    ),
    pytest.param(
        "cqr_cate",
        "code/cqr_cate/CQR_causal_inference.ipynb",
        "_posts/2023-07-17-cqr-cate.md",
        id="cqr_cate",
    ),
    pytest.param(
        "threshold_dependent_opt",
        "code/threshold_dependent_opt/blog_threshold_dependent_opt.ipynb",
        "_posts/2023-01-06-threshold-dependent-opt.md",
        id="threshold_dependent_opt",
    ),
    pytest.param(
        "boruta",
        "code/boruta/blog_datalab_boruta.ipynb",
        "_posts/2022-09-05-boruta.md",
        id="boruta",
    ),
    pytest.param(
        "conditional_density_estimation",
        "code/conditional_density_estimation/cde.ipynb",
        "_posts/2023-06-16-conditional-density-estimation.md",
        id="conditional_density_estimation",
    ),
    pytest.param(
        "distance_metrics",
        "code/distance_metrics/blog_distance_metrics_EN.ipynb",
        "_posts/2020-07-05-distancia.md",
        id="distance_metrics",
    ),
]


def _build(slug: str, nb_relpath: str):
    nb_path = REPO_ROOT / nb_relpath
    assert nb_path.exists(), f"missing notebook fixture: {nb_path}"
    text, images = n.build_post(nb_path, slug)
    return text, images


@pytest.mark.parametrize("slug,nb_relpath,post_relpath", STRICT_FIXTURES)
def test_code_blocks_match_published(slug, nb_relpath, post_relpath):
    generated, _ = _build(slug, nb_relpath)
    committed = (REPO_ROOT / post_relpath).read_text(encoding="utf-8")
    assert n.extract_code_blocks(generated) == n.extract_code_blocks(committed)


@pytest.mark.parametrize("slug,nb_relpath,post_relpath", STRICT_FIXTURES)
def test_output_blocks_match_published(slug, nb_relpath, post_relpath):
    generated, _ = _build(slug, nb_relpath)
    committed = (REPO_ROOT / post_relpath).read_text(encoding="utf-8")
    assert n.extract_output_blocks(generated) == n.extract_output_blocks(committed)


@pytest.mark.parametrize("slug,nb_relpath,post_relpath", STRICT_FIXTURES)
def test_html_tables_match_published(slug, nb_relpath, post_relpath):
    generated, _ = _build(slug, nb_relpath)
    committed = (REPO_ROOT / post_relpath).read_text(encoding="utf-8")
    assert n.extract_html_tables(generated) == n.extract_html_tables(committed)


@pytest.mark.parametrize("slug,nb_relpath,post_relpath", STRICT_FIXTURES)
def test_image_embeds_match_published(slug, nb_relpath, post_relpath):
    generated, images = _build(slug, nb_relpath)
    committed = (REPO_ROOT / post_relpath).read_text(encoding="utf-8")
    # Same figures referenced, in the same order.
    assert n.extract_image_names(generated) == n.extract_image_names(committed)
    # And every figure is embedded with the exact div-align template in both.
    for fname in n.extract_image_names(generated):
        embed = n.embed_for(slug, fname)
        assert embed in generated, f"generated post missing embed for {fname}"
        assert embed in committed, f"published post missing embed for {fname}"


# --------------------------------------------------------------------------- #
# Unit tests for the format primitives and directive engine.
# --------------------------------------------------------------------------- #

def test_embed_uses_div_align_template():
    assert n.embed_for("foo", "output_1_0.png") == (
        '<p><div align="justify"><center><img '
        'src="{{ site.baseurl }}/assets/img/foo/output_1_0.png"></center></div></p>'
    )


def test_output_indentation_is_four_spaces():
    md = "```python\nprint(1)\n```\n\n    1\n"
    assert n.extract_output_blocks(md) == ["1"]


def _code_cell(source, outputs=None):
    import nbformat

    cell = nbformat.v4.new_code_cell(source)
    cell["outputs"] = outputs or []
    return cell


def _nb(cells):
    import nbformat

    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    return nb


def test_skip_directive_drops_cell_but_keeps_index():
    nb = _nb([_code_cell("a = 1"), _code_cell("# nb2post: skip\nsecret = 2"), _code_cell("b = 3")])
    n.apply_directives(nb)
    code_sources = [c["source"] for c in nb.cells if c["cell_type"] == "code"]
    assert code_sources == ["a = 1", "b = 3"]
    # placeholder preserved -> three cells total, so figure numbering stays stable
    assert len(nb.cells) == 3


def test_merge_directive_joins_into_previous_block():
    nb = _nb([_code_cell("def f():\n    pass"), _code_cell("# nb2post: merge\nf()")])
    n.apply_directives(nb)
    code_sources = [c["source"] for c in nb.cells if c["cell_type"] == "code"]
    assert code_sources == ["def f():\n    pass\n\nf()"]
    assert len(nb.cells) == 2  # second slot kept as placeholder


def test_skip_output_directive_keeps_code_drops_output():
    nb = _nb([_code_cell("# nb2post: skip-output\nx = 1", outputs=[{"output_type": "stream"}])])
    n.apply_directives(nb)
    cell = nb.cells[0]
    assert cell["source"] == "x = 1"
    assert cell["outputs"] == []
