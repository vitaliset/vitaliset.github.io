# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "nbconvert>=6,<8",
#     "nbformat>=5",
# ]
# ///
"""Scaffold a Jekyll post from a Jupyter notebook.

This is a *scaffolding* tool, not a full publisher. It runs ``nbconvert`` over a
notebook and produces:

* code cells as ```python fenced blocks (no indentation),
* cell stdout / text outputs as 4-space-indented blocks,
* figures saved to ``assets/img/<slug>/output_<cell>_<out>.png`` and embedded as
  ``<p><div align="justify"><center><img src="{{ site.baseurl }}/assets/img/<slug>/output_X_Y.png"></center></div></p>``,
* a front-matter header and an "experiments" link pointing at ``code/<slug>``.

The prose pulled from markdown cells comes through as plain markdown. The
published posts wrap prose in justified HTML (``<p><div align="justify">...</div></p>``)
which nbconvert cannot reproduce, so **prose is hand-authored afterward**. The
job of this script is to remove the mechanical steps (image extraction, naming,
path rewriting, front-matter boilerplate) and to stay byte-consistent with the
recent posts for the code/output/image blocks (see scripts/tests/).

Usage (recommended, zero-setup via uv):

    uv run scripts/notebook_to_post.py \
        --notebook code/<slug>/<file>.ipynb --slug <slug> --date YYYY-MM-DD \
        [--title "..."] [--category "🇺🇸, basic"] [--description "..."]

Or with a plain Python env (pip install -r scripts/requirements.txt):

    python scripts/notebook_to_post.py --notebook ... --slug ... --date ...
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_BASE = "https://github.com/vitaliset/vitaliset.github.io/tree/master/code"

# Matches the trailing "_<cell>_<out>.<ext>" that nbconvert appends to figure names.
_IMG_SUFFIX = re.compile(r"_(\d+)_(\d+)\.(png|jpe?g|gif|svg)$", re.IGNORECASE)
# Markdown image reference, e.g. ![png](output_files/output_17_0.png)
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+\.(?:png|jpe?g|gif|svg))\)", re.IGNORECASE)
_FENCE = re.compile(r"^\s*```")


def _normalized_image_name(raw_name: str) -> str:
    """Return output_<cell>_<out>.<ext> regardless of nbconvert's unique_key."""
    base = Path(raw_name).name
    m = _IMG_SUFFIX.search(base)
    if not m:
        return base
    cell, out, ext = m.groups()
    return f"output_{cell}_{out}.{ext.lower()}"


# A cell whose first line is a directive comment is handled specially. The
# directive line itself is stripped from the rendered post (it stays in the
# notebook, which remains the single source of truth).
_DIRECTIVE = re.compile(r"^#\s*nb2post:\s*([a-z][a-z-]*)\s*$", re.IGNORECASE)

# Tag applied to a `skip-input` cell so nbconvert's TagRemovePreprocessor drops
# the input (code) while keeping the output (figure/table) and the cell's index.
SKIP_INPUT_TAG = "nb2post-skip-input"


def _cell_directive(source: str) -> Tuple[str | None, str]:
    """Return (directive, source_without_directive_line) for a code cell."""
    lines = source.split("\n")
    if lines:
        m = _DIRECTIVE.match(lines[0].strip())
        if m:
            return m.group(1).lower(), "\n".join(lines[1:]).lstrip("\n")
    return None, source


def apply_directives(nb) -> None:
    """Rewrite notebook cells in place to honor nb2post directives.

    * ``skip``        -> drop the cell (code + outputs) from the rendered post.
    * ``skip-input``  -> hide the code but keep its output (e.g. show only a
                         figure or table, not the plotting boilerplate).
    * ``skip-output`` -> keep the code, drop its outputs.
    * ``merge``       -> append this cell's code/outputs to the previous code cell
                         so they render as a single ```python block.

    A consumed cell (skip / the source of a merge) is replaced by an empty
    placeholder rather than deleted, so the *absolute cell index* of every
    remaining cell is preserved. nbconvert names extracted figures
    ``output_<cell_index>_<out>.png`` from that index, so preserving it keeps
    figure filenames stable (matching the already-published posts) no matter how
    many cells a directive removes. ``skip-input`` keeps the cell in place (it
    just tags it for TagRemovePreprocessor), so it is index-stable too.
    """
    import nbformat

    def blank():
        return nbformat.v4.new_markdown_cell("")

    new_cells = []
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            new_cells.append(cell)
            continue
        directive, body = _cell_directive(cell.get("source", ""))
        if directive == "skip":
            new_cells.append(blank())
            continue
        if directive == "skip-input":
            cell["source"] = body
            cell.setdefault("metadata", {}).setdefault("tags", [])
            if SKIP_INPUT_TAG not in cell["metadata"]["tags"]:
                cell["metadata"]["tags"].append(SKIP_INPUT_TAG)
            new_cells.append(cell)
            continue
        if directive == "skip-output":
            cell["source"] = body
            cell["outputs"] = []
            new_cells.append(cell)
            continue
        if directive == "merge":
            prev = next((c for c in reversed(new_cells) if c.get("cell_type") == "code"), None)
            if prev is None:
                print("  warning: 'merge' directive on a cell with no preceding code cell; treating as normal")
                cell["source"] = body
                new_cells.append(cell)
                continue
            prev["source"] = prev["source"].rstrip("\n") + "\n\n" + body
            prev["outputs"] = list(prev.get("outputs", [])) + list(cell.get("outputs", []))
            new_cells.append(blank())  # keep the index slot so figure numbers stay stable
            continue
        new_cells.append(cell)
    nb.cells = new_cells


def convert_notebook(nb_path: Path) -> Tuple[str, Dict[str, bytes]]:
    """Run nbconvert (after applying directives) and return (markdown_body, images)."""
    import nbformat  # imported lazily so --help works without deps
    from nbconvert import MarkdownExporter
    from nbconvert.preprocessors import TagRemovePreprocessor

    nb = nbformat.read(str(nb_path), as_version=4)
    apply_directives(nb)

    exporter = MarkdownExporter()
    # Honor the skip-input tag set by apply_directives: hide the code, keep output.
    # This keeps the cell in place, so figure indices stay stable.
    tag_remove = TagRemovePreprocessor(remove_input_tags={SKIP_INPUT_TAG})
    tag_remove.enabled = True
    exporter.register_preprocessor(tag_remove, enabled=True)

    body, resources = exporter.from_notebook_node(
        nb, resources={"unique_key": "output", "output_files_dir": "output_files"}
    )
    images: Dict[str, bytes] = {}
    for key, data in (resources.get("outputs") or {}).items():
        images[_normalized_image_name(key)] = data
    return body, images


def embed_for(slug: str, fname: str) -> str:
    return (
        '<p><div align="justify"><center><img src="{{ site.baseurl }}/assets/img/'
        + slug
        + "/"
        + fname
        + '"></center></div></p>'
    )


def rewrite_images(body: str, slug: str) -> str:
    """Replace nbconvert ![alt](path) image refs with the centered <img> embed."""

    def repl(match: "re.Match[str]") -> str:
        fname = _normalized_image_name(match.group(1))
        return embed_for(slug, fname)

    return _MD_IMAGE.sub(repl, body)


def normalize_blank_lines(body: str) -> str:
    """Collapse whitespace-only lines to empty (outside code fences).

    nbconvert pads stream outputs and figures with 4-space lines (e.g. ``"    "``);
    the published posts use truly blank lines. We normalize so generated output is
    consistent with the recent posts. Lines inside ```...``` fences are untouched.
    """
    out: List[str] = []
    in_fence = False
    for line in body.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if not in_fence and line.strip() == "":
            out.append("")
        else:
            out.append(line)
    text = "\n".join(out)
    # collapse 3+ consecutive blank lines into a single blank line
    return re.sub(r"\n{3,}", "\n\n", text)


def build_front_matter(slug: str, title: str, category: str, description: str) -> str:
    return (
        "---\n"
        "layout: post\n"
        f"title: {title}\n"
        f"featured-img: {slug}\n"
        f"category: [{category}]\n"
        "mathjax: true\n"
        f"description: {description}\n"
        "---\n"
    )


def experiments_paragraph(slug: str) -> str:
    url = f"{EXPERIMENTS_BASE}/{slug}"
    return (
        '<p><div align="justify">You can find all files and environments for '
        f'reproducing the experiments in the <a href="{url}">repository of this '
        "post</a>.</div></p>\n"
    )


SCAFFOLD_BANNER = (
    "<!-- SCAFFOLD generated by scripts/notebook_to_post.py. Prose below comes from "
    "the notebook's markdown cells as plain markdown; wrap it in "
    '<p><div align="justify">...</div></p> to match the site style, then delete this '
    "comment before publishing. -->\n"
)


def build_post(
    nb_path: Path,
    slug: str,
    *,
    title: str | None = None,
    category: str = "🇺🇸, basic",
    description: str = "TODO: one-line description.",
    include_banner: bool = True,
    include_experiments_link: bool = True,
) -> Tuple[str, Dict[str, bytes]]:
    """Build the post markdown (not written to disk) and collect figure bytes."""
    body, images = convert_notebook(nb_path)
    body = rewrite_images(body, slug)
    body = normalize_blank_lines(body).strip("\n")

    parts = [build_front_matter(slug, title or slug, category, description)]
    if include_banner:
        parts.append("\n" + SCAFFOLD_BANNER)
    parts.append("\n" + body + "\n")
    if include_experiments_link:
        parts.append("\n___\n\n" + experiments_paragraph(slug))
    return "".join(parts), images


# --------------------------------------------------------------------------- #
# Extraction helpers — shared with the snapshot tests in scripts/tests/.
# --------------------------------------------------------------------------- #

def extract_code_blocks(md: str) -> List[str]:
    """Return the contents of every ```python fenced block, in order."""
    return [b.rstrip("\n") for b in re.findall(r"```python\n(.*?)\n```", md, flags=re.DOTALL)]


def extract_output_blocks(md: str) -> List[str]:
    """Return the dedented text outputs of code cells, in order.

    Output is **fence-aware**: nbconvert renders a code cell's stdout/text result
    as 4-space-indented lines *immediately after* the cell's closing ``` fence. We
    only capture indented runs in that post-code zone, which ends at the next
    non-indented line (prose, an HTML table ``<div>``, an image embed, or another
    fence). This deliberately ignores 4-space-indented *markdown prose* such as
    nested bullet lists, which are hand-authored and must not be compared.
    Trailing whitespace-only lines are dropped (an nbconvert ``"    "`` artifact).
    """
    blocks: List[str] = []
    current: List[str] = []
    in_fence = False
    after_code = False

    def flush() -> None:
        if current:
            blocks.append("\n".join(current).rstrip("\n"))
            current.clear()

    for line in md.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
            flush()
            after_code = not in_fence  # True right after a fence closes
            continue
        if in_fence or not after_code:
            continue
        if line.strip() == "":
            flush()  # end this run, but stay in the cell's output zone
        elif line.startswith("    "):
            current.append(line[4:])
        else:
            flush()
            after_code = False  # prose / table / image ends the output zone
    flush()
    return blocks


def extract_html_tables(md: str) -> List[str]:
    """Return the nbconvert-rendered pandas DataFrame tables (``<div>…</div>``)."""
    lines = md.split("\n")
    blocks: List[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "<div>":
            end = next((k for k in range(i, len(lines)) if lines[k].strip() == "</div>"), None)
            if end is not None:
                block = lines[i:end + 1]
                if any('class="dataframe"' in x for x in block):
                    blocks.append("\n".join(block))
                i = end + 1
                continue
        i += 1
    return blocks


def extract_image_names(md: str) -> List[str]:
    """Return the output_<cell>_<out>.<ext> figure names referenced, in order."""
    return re.findall(r"output_\d+_\d+\.(?:png|jpe?g|gif|svg)", md, flags=re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Prose extraction — used by the prose-sync test to enforce that a post's prose
# and its notebook's markdown stay in sync (two views of the same text).
# --------------------------------------------------------------------------- #
import html as _html

# Boilerplate footer paragraph(s) that exist only in the published post.
_PROSE_FOOTER = (
    "repositório deste post",
    "repository of this post",
    "repositório de experimentos",
    "originalmente publicado no Medium",
)
_FENCE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_DISPLAY_MATH = re.compile(r"\$\$.*?\$\$", re.DOTALL)  # standalone in posts, inline-in-cell in notebooks
_EMPHASIS_US = re.compile(r"(?<!\w)_|_(?!\w)")  # emphasis underscores, not intra-word (rand_score)


def _canonical_prose(text: str) -> str:
    """Reduce a prose fragment (HTML or markdown) to comparable plain text.

    Normalization is about *consistency* between the two formats, not pretty
    output: decode entities, turn links into their label, drop tags, and remove
    inline-formatting markers (`*`, backtick, blockquote `>`, heading `#`, and
    emphasis underscores) so the post's HTML and the notebook's markdown collapse
    to the same string.
    """
    text = _html.unescape(text)
    text = re.sub(r"<a [^>]*>(.*?)</a>", r"\1", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # markdown links (label has no ])
    text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)  # real HTML tags only (not math '<'/'>')
    text = text.replace(r"\left", "").replace(r"\right", "")  # LaTeX sizing: \left( == (
    text = text.replace("`", "").replace("*", "").replace("<", "").replace(">", "").replace("#", "")
    text = _EMPHASIS_US.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def post_prose_text(post_md: str) -> str:
    """Canonical joined prose of a post (the justified-text paragraphs)."""
    out = []
    # Body prose is <p><div align="justify">…</div></p>; bibliography entries are
    # often <p align="justify">…</p> (no inner div). Capture both in document order.
    pattern = r'<p><div align="justify">(.*?)</div></p>|<p align="justify">(.*?)</p>'
    for m in re.finditer(pattern, post_md, flags=re.DOTALL):
        p = m.group(1) if m.group(1) is not None else m.group(2)
        if "<img" in p:
            continue
        c = _canonical_prose(p)
        if c and not any(f in c for f in _PROSE_FOOTER):
            out.append(c)
    return " ".join(out)


def notebook_prose_text(nb_path) -> str:
    """Canonical joined prose of a notebook's markdown cells.

    Skips headings, tables, horizontal rules, image/caption lines, fenced code,
    and the post-only footer, so the result is comparable to ``post_prose_text``.
    """
    import nbformat

    nb = nbformat.read(str(nb_path), as_version=4)
    out = []
    for cell in nb.cells:
        if cell.get("cell_type") != "markdown":
            continue
        src = _FENCE_BLOCK.sub("", cell.get("source", ""))
        src = _DISPLAY_MATH.sub("", src)
        for block in src.split("\n\n"):
            s = block.strip()
            if not s or s.startswith("#") or s.startswith("|"):
                continue
            if re.fullmatch(r"[-_*]{3,}", s):  # horizontal rule
                continue
            if "![" in s or "<img" in s or s.startswith("<center>"):
                continue
            block = re.sub(r"(?m)^\s*(?:\d+\.|[-*+])\s+", "", block)  # list markers
            c = _canonical_prose(block)
            if c and not any(f in c for f in _PROSE_FOOTER):
                out.append(c)
    return " ".join(out)


def write_post(post_text: str, images: Dict[str, bytes], slug: str, date: str) -> Path:
    img_dir = REPO_ROOT / "assets" / "img" / slug
    img_dir.mkdir(parents=True, exist_ok=True)
    for fname, data in images.items():
        target = img_dir / fname
        if target.exists():
            print(f"  warning: overwriting existing image {target.relative_to(REPO_ROOT)}")
        target.write_bytes(data)

    post_path = REPO_ROOT / "_posts" / f"{date}-{slug}.md"
    post_path.write_text(post_text, encoding="utf-8")
    return post_path


def main() -> None:
    # Emoji in help/category (e.g. 🇺🇸) crash on Windows' default cp1252 console;
    # force UTF-8 so --help and unicode prints work everywhere.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook", required=True, type=Path, help="path to the .ipynb file")
    parser.add_argument("--slug", required=True, help="post slug, e.g. evaluating_ranking_in_regression")
    parser.add_argument("--date", required=True, help="post date YYYY-MM-DD")
    parser.add_argument("--title", default=None, help="post title (defaults to the slug)")
    parser.add_argument("--category", default="🇺🇸, basic", help='front-matter category list, e.g. "🇺🇸, basic"')
    parser.add_argument("--description", default="TODO: one-line description.", help="front-matter description (one-liner; used as the meta description and homepage card blurb)")
    args = parser.parse_args()

    nb_path = args.notebook if args.notebook.is_absolute() else (Path.cwd() / args.notebook)
    if not nb_path.exists():
        raise SystemExit(f"notebook not found: {nb_path}")

    post_text, images = build_post(
        nb_path, args.slug, title=args.title, category=args.category, description=args.description
    )
    post_path = write_post(post_text, images, args.slug, args.date)
    print(f"Wrote {post_path.relative_to(REPO_ROOT)} and {len(images)} image(s) to assets/img/{args.slug}/")
    print("Next: hand-author the prose (wrap in justified HTML) before publishing.")


if __name__ == "__main__":
    main()
