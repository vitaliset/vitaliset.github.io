"""Availability-note invariants for the EN/PT language system.

The post/page layouts render a "this content is only available in <lang>" note
on *monolingual* content and suppress it on bilingual content:

  - post.html: note iff (not `bilingual: true`) and the categories contain a
    single 🇺🇸/🇧🇷 flag.
  - page.html: note iff (not `bilingual: true`) and the page has an explicit
    `lang:` field.

The note is what tells an EN reader that a Portuguese-only post/page (or vice
versa) is not available in their language. Translating the content is supposed
to remove it: you add both EN and PT `.i18n` blocks (inline or via an include)
and set `bilingual: true`, which suppresses the note.

The failure this guards against: translating the *body* but forgetting
`bilingual: true`. Then the stale "only available in X" note keeps rendering
under real bilingual content. These tests parse the source front matter + body
(no Jekyll build needed) and assert the note logic stays consistent with the
content actually present.
"""
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INCLUDES_DIR = REPO_ROOT / "_includes"

# An .i18n content block in a given language, tolerant of attribute order.
_I18N = {
    "en": (re.compile(r'class="i18n"\s+lang="en"'), re.compile(r'lang="en"\s+class="i18n"')),
    "pt": (re.compile(r'class="i18n"\s+lang="pt"'), re.compile(r'lang="pt"\s+class="i18n"')),
}
_INCLUDE = re.compile(r"{%\s*include\s+([^\s%]+)\s*%}")


def _split_front_matter(text):
    """Return (front_matter, body). Files without front matter yield ("", text)."""
    if not text.startswith("---"):
        return "", text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return "", text
    return parts[1], parts[2]


def _expand_includes(body):
    """Append the text of any {% include X %} so in-include .i18n blocks count
    (e.g. the About page is `{% include about-body.html %}`)."""
    chunks = [body]
    for name in _INCLUDE.findall(body):
        inc = INCLUDES_DIR / name
        if inc.exists():
            chunks.append(inc.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def _has_block(body, lang):
    return any(p.search(body) for p in _I18N[lang])


class Content:
    """A post or page, classified by its language front matter and body."""

    def __init__(self, path):
        self.path = path
        self.id = f"{path.parent.name}/{path.name}"
        text = path.read_text(encoding="utf-8")
        fm, body = _split_front_matter(text)
        body = _expand_includes(body)

        self.is_post = path.parent.name == "_posts"
        self.bilingual = bool(re.search(r"^\s*bilingual:\s*true\s*$", fm, re.M))
        self.has_lang = bool(re.search(r"^\s*lang:\s*\S", fm, re.M))
        self.has_title_pt = bool(re.search(r"^\s*title_pt:\s*\S", fm, re.M))
        self.has_us_flag = "\U0001F1FA\U0001F1F8" in fm  # 🇺🇸
        self.has_br_flag = "\U0001F1E7\U0001F1F7" in fm  # 🇧🇷
        self.inline_en = _has_block(body, "en")
        self.inline_pt = _has_block(body, "pt")

    @property
    def inline_bilingual(self):
        """The body carries both EN and PT content blocks."""
        return self.inline_en and self.inline_pt

    @property
    def note_renders(self):
        """Mirror of the layout logic: would the availability note be emitted?"""
        if self.bilingual:
            return False
        if self.is_post:
            return self.has_us_flag or self.has_br_flag
        return self.has_lang


def _content_files():
    files = list((REPO_ROOT / "_posts").glob("*.md"))
    files += sorted((REPO_ROOT / "_pages").glob("*.md"))
    root_404 = REPO_ROOT / "404.md"
    if root_404.exists():
        files.append(root_404)
    return files


CONTENT = [Content(p) for p in _content_files()]
PARAMS = [pytest.param(c, id=c.id) for c in CONTENT]


@pytest.mark.parametrize("c", PARAMS)
def test_translated_body_is_marked_bilingual(c):
    """In-body EN+PT blocks must come with `bilingual: true`.

    Otherwise the "only available in X" note keeps rendering on top of content
    that is, in fact, fully translated.
    """
    if c.inline_bilingual:
        assert c.bilingual, (
            f"{c.id} has both EN and PT .i18n blocks but is not `bilingual: true`; "
            f"the 'only available in X' note would still render under translated content."
        )


@pytest.mark.parametrize("c", PARAMS)
def test_bilingual_has_real_translation(c):
    """`bilingual: true` must correspond to actual bilingual content.

    Either a `title_pt` (the post/page title flips) or both-language body blocks.
    A hollow flag would silently suppress the note with nothing to switch to.
    """
    if c.bilingual:
        assert c.has_title_pt or c.inline_bilingual, (
            f"{c.id} is `bilingual: true` but has neither title_pt nor both-language "
            f"body blocks."
        )


@pytest.mark.parametrize("c", PARAMS)
def test_no_availability_note_on_bilingual(c):
    """Bilingual content never shows the monolingual availability note."""
    if c.bilingual:
        assert not c.note_renders, f"{c.id} is bilingual yet would render an availability note."


def test_note_mechanism_is_exercised():
    """Guard against the suite passing on a mis-parsed/empty set: the repo must
    still contain at least one monolingual file that shows the note and one
    bilingual file that hides it."""
    assert any(c.note_renders for c in CONTENT), "no content renders the availability note"
    assert any(c.bilingual for c in CONTENT), "no bilingual content found"
