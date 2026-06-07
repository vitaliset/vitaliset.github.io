# vitaliset.github.io

Personal blog for technical discussions on data science! The layout is a small
adaptation of the <a href="https://janczizikow.github.io/sleek/">Sleek</a> theme.

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
_pages/      Standalone pages (about, sobre, links, interview, etc. — each with its permalink)
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

## Credits

- Logo / favicon: chicken from [OpenMoji](https://openmoji.org/), **modified** (cropped the
  `viewBox` to frame the icon better), licensed under
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). As a derivative work under
  _ShareAlike_, the icon (`_includes/logo.svg` and `assets/img/icons/favicon.svg`) remains
  under CC BY-SA 4.0.
