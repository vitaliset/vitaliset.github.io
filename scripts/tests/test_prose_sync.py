"""Prose-sync tests: a post's prose and its notebook's markdown must agree.

The published post and its notebook are two views of the same writing. The
snapshot tests (test_notebook_to_post.py) already lock the code/outputs/tables/
images; these tests lock the *prose*: editing one without the other is caught.

Comparison is on canonical joined prose (see notebook_to_post.post_prose_text /
notebook_prose_text): HTML vs markdown formatting, link syntax, emphasis markers,
headings, display equations, captions, and the post-only footer are normalized
away, so what's compared is the actual wording (including inline math).
"""
from pathlib import Path

import pytest

import notebook_to_post as n

REPO_ROOT = Path(__file__).resolve().parents[2]

# (slug, notebook path, post path) for every notebook-backed post.
PROSE_FIXTURES = [
    pytest.param("r_squared", "code/r_squared/blog_r2_score_EN.ipynb",
                 "_posts/2023-10-12-r2-score.md", id="r_squared"),
    pytest.param("evaluating_ranking_in_regression",
                 "code/evaluating_ranking_in_regression/blog_evaluating_ranking_in_regression.ipynb",
                 "_posts/2024-11-17-evaluating-ranking-in-regression.md", id="evaluating_ranking_in_regression"),
    pytest.param("metakmeans", "code/metakmeans/blog_metakmeans.ipynb",
                 "_posts/2022-10-23-metakmeans.md", id="metakmeans"),
    pytest.param("covariate_introduction",
                 "code/covariate_introduction/covariate_shift_introduction_EN.ipynb",
                 "_posts/2020-08-02-covariate-shift-0-introduction.md", id="covariate_introduction"),
    pytest.param("cqr_cate", "code/cqr_cate/CQR_causal_inference.ipynb",
                 "_posts/2023-07-17-cqr-cate.md", id="cqr_cate"),
    pytest.param("threshold_dependent_opt",
                 "code/threshold_dependent_opt/blog_threshold_dependent_opt.ipynb",
                 "_posts/2023-01-06-threshold-dependent-opt.md", id="threshold_dependent_opt"),
    pytest.param("boruta", "code/boruta/blog_datalab_boruta.ipynb",
                 "_posts/2022-09-05-boruta.md", id="boruta"),
    pytest.param("conditional_density_estimation", "code/conditional_density_estimation/cde.ipynb",
                 "_posts/2023-06-16-conditional-density-estimation.md", id="conditional_density_estimation"),
]


@pytest.mark.parametrize("slug,nb_relpath,post_relpath", PROSE_FIXTURES)
def test_post_prose_matches_notebook(slug, nb_relpath, post_relpath):
    nb_path = REPO_ROOT / nb_relpath
    post_md = (REPO_ROOT / post_relpath).read_text(encoding="utf-8")
    post = n.post_prose_text(post_md)
    notebook = n.notebook_prose_text(nb_path)
    assert post == notebook
