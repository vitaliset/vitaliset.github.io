# Development

How to run, build, and work on [vitaliset.github.io](https://vitaliset.github.io/).

## Running locally

To preview the site before publishing (instead of testing live):

**Setup (one time only):**

1. Install Ruby with the DevKit and finish the toolchain (press Enter at the menu):
   ```powershell
   winget install RubyInstallerTeam.RubyWithDevKit.3.1
   ridk install
   ```
   Open a new terminal and confirm: `ruby -v`.
2. Install Bundler: `gem install bundler`.
3. In the site folder, install the dependencies: `bundle install`.

> The `Gemfile` already includes `webrick` (the server required on Ruby 3.x) and
> `kramdown-parser-gfm` (the GFM parser, split out of kramdown 2.x) — just run the
> `bundle install` above.

**Day to day:**

```powershell
bundle exec jekyll serve --livereload
```

Open http://localhost:4000. The site reloads on save. Stop it with `Ctrl+C`.

> Changes to `_config.yml` are not reloaded automatically — stop and rerun the command.

## Repository structure

```
_posts/      Blog posts (markdown, named YYYY-MM-DD-slug.md)
_pages/      Standalone pages (about, links, interview, etc. — each with its permalink)
_layouts/    Jekyll templates (default, page, post)
_includes/   Partials (head, header, footer, mathjax, ...)
_sass/       Style source (SCSS) — see the note below
_js/         JavaScript source — see the note below
assets/      Served static files: compiled CSS/JS, images, icons
files/       PDFs (resume, transcripts)
code/        Per-post notebooks and environments (reproducible source; excluded from the build)
scripts/     notebook -> post pipeline (see scripts/README.md)
```

To generate a post from a notebook in `code/<slug>/`, use the pipeline documented in
[`scripts/README.md`](scripts/README.md).

## Compiled styles and assets

`assets/css/main.css` and `assets/js/bundle.js` are **versioned artifacts** — they are what
the site actually serves. **Do not edit them by hand** — change the source and recompile.

The original theme shipped a Gulp/npm toolchain to build these. It was already broken
(gulp-3 syntax under gulp 4, never installed) and it carried a large pile of npm
vulnerabilities, so it has been **removed**. What remains:

**Styles (`_sass/` → `main.css`):** recompile with the Sass that ships with the Ruby toolchain:

```powershell
sass --scss --style compressed --sourcemap=none _sass/jekyll-sleek.scss assets/css/main.css
```

> The only practical difference from the old Gulp build is that this does not add vendor
> prefixes (autoprefixer) — irrelevant for current browsers.

**JS (`bundle.js`):** now a **vendored artifact** (jQuery + velocity + lazysizes, bundled).
Its source is [`_js/scripts.js`](_js/scripts.js), kept for reference, but there is no JS
build wired up anymore. The JS rarely changes; if it ever needs to, set up a small modern
bundler (e.g. [esbuild](https://esbuild.github.io/)) to rebuild `bundle.js` from `_js/`.

## Bilingual pages (EN/PT)

The site supports per-page English/Portuguese content with an instant in-place toggle —
the `EN | PT` pill in the header. Switching cross-fades the content with **no page reload**
and remembers the choice in `localStorage`. The pill only appears on pages with
`bilingual: true` in their front matter (so monolingual pages don't show a dead toggle); a
remembered choice still applies the next time you land on a bilingual page.

**How it works.** Everything lives in [`_includes/i18n.html`](_includes/i18n.html), loaded
high in `<head>`. It has three parts:

1. A tiny script that resolves the language *before paint* — `localStorage['vs-lang']`, else
   the page's `<html lang>` (`pt*` ⇒ `pt`), else `en` — and sets `html[data-lang="…"]`. This
   prevents any flash of the wrong language.
2. Inline CSS that shows only the block matching the active language. English is the default,
   so **no-JS visitors and search engines always see English**; Portuguese is revealed via
   `display: revert` when active. The pill's active segment is styled purely from
   `html[data-lang]`, so it looks right before any JS runs.
3. A delegated click handler that cross-fades (`.i18n-group.is-fading`), flips `data-lang`,
   and persists the choice. It honours `prefers-reduced-motion`.

> These styles are intentionally **inline in the head include** (not in `main.css`): the
> no-flash rules must be present before first paint. A mirror of the pill styles also lives in
> [`_sass/components/_lang-switch.scss`](_sass/components/_lang-switch.scss) for consistency if
> `main.css` is ever recompiled — keep the two in sync.

**Making a page bilingual.** Wrap each language's content in an `.i18n` block inside an
`.i18n-group` (the group is what fades):

```html
<div class="i18n-group">
  <div class="i18n" lang="en"> …English… </div>
  <div class="i18n" lang="pt"> …Português… </div>
</div>
```

Untranslated content needs no wrapper — it simply shows in both languages (English fallback).
The About page is the reference implementation: it's a single bilingual URL (`/about/`) whose
two languages live in [`_includes/about-body.html`](_includes/about-body.html), included by
[`_pages/about.md`](_pages/about.md) (`lang: en-US`). Set `bilingual: true` in front matter to
show the header pill, and optionally `title_pt:` / `title_en:` so the hero title switches too
(handled in [`_layouts/page.html`](_layouts/page.html)).

### Bilingual posts

Posts use the **same toggle** but keep a **single URL** (e.g. `/covariate-shift-0-introduction/`):
the **code, outputs and figures are shared** (shown in both languages) and only the **prose**
swaps. There's no `.i18n-group`, so the swap is instant (no fade — better for long posts). The
reference implementation is [`_posts/2020-08-02-covariate-shift-0-introduction.md`](_posts/2020-08-02-covariate-shift-0-introduction.md).

Front matter: `bilingual: true`, `lang: en-US`, `title_pt:`, and keep `category: [🇺🇸, 🇧🇷, …]`.
Then, for each prose paragraph (and each section heading), author an EN/PT pair; leave code
fences, output blocks and `<img>` embeds **unwrapped** (shared). Use `markdown="1"` whenever the
wrapped block contains markdown (e.g. a `##` heading or a list); pure-HTML prose doesn't need it:

```html
<div class="i18n" lang="en"><p><div align="justify">… English …</div></p></div>
<div class="i18n" lang="pt"><p><div align="justify">… Português …</div></p></div>

<div class="i18n" lang="en" markdown="1">
## Example of dataset shift
</div>
<div class="i18n" lang="pt" markdown="1">
## Exemplo de dataset shift
</div>
```

**Sources of truth & guarantees.** The per-language **notebooks** under [`code/<slug>/`](code)
(`*_EN.ipynb` + `*_PT*.ipynb`) remain canonical; the post is the assembled bilingual view. Two
test layers keep it honest (run `pytest scripts/tests/`):
- [`test_prose_sync.py`](scripts/tests/test_prose_sync.py) — each language track of the post
  matches its own notebook. It uses `notebook_to_post.strip_i18n_other(md, lang)` to compare one
  language at a time; for monolingual posts that's a no-op, so they're unaffected.
- [`test_notebook_parity.py`](scripts/tests/test_notebook_parity.py) — the EN and PT notebooks of
  a pair stay parallel, compared at the **paragraph/token level** (not cell-by-cell, since a cell
  may legitimately be split differently between languages): identical flattened sequence of code
  blocks / headings / prose paragraphs / figures, with code byte-identical. This is what lets
  code/figures be shared and prose pair 1:1. To add a bilingual post, register its notebook pair
  here and its two language fixtures in `test_prose_sync.py`.

The reference bilingual posts are `covariate_introduction` and `r_squared`. If a pair ever does
drift (a paragraph added on one side, a figure only on one), the parity test fails — reconcile the
two notebooks before regenerating the post.
