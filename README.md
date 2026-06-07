# vitaliset.github.io

Blog pessoal para discussões técnicas de ciência de dados! O layout é uma pequena adaptação do tema <a href="https://janczizikow.github.io/sleek/">Sleek</a>.

## Rodando localmente

Para pré-visualizar o site antes de publicar (em vez de testar direto no ar):

**Setup (uma vez só):**

1. Instale o Ruby com o DevKit e finalize o toolchain (aperte Enter no menu):
   ```powershell
   winget install RubyInstallerTeam.RubyWithDevKit.3.1
   ridk install
   ```
   Abra um novo terminal e confirme: `ruby -v`.
2. Instale o Bundler: `gem install bundler`.
3. Na pasta do site, instale as dependências: `bundle install`.

> O `Gemfile` já inclui `webrick` (servidor necessário no Ruby 3.x) e `kramdown-parser-gfm` (parser GFM, separado do kramdown 2.x) — só rodar o `bundle install` acima.

**No dia a dia:**

```powershell
bundle exec jekyll serve --livereload
```

Abra http://localhost:4000. O site recarrega sozinho ao salvar. Pare com `Ctrl+C`.

> Mudanças no `_config.yml` não recarregam automaticamente — pare e rode o comando de novo.

## Estrutura do repositório

```
_posts/      Posts do blog (markdown, nomeados YYYY-MM-DD-slug.md)
_pages/      Páginas avulsas (about, sobre, links, interview, etc. — cada uma com seu permalink)
_layouts/    Templates do Jekyll (default, page, post)
_includes/   Partials (head, header, footer, mathjax, ...)
_sass/       Fonte dos estilos (SCSS) — veja a nota abaixo
_js/         Fonte do JavaScript — veja a nota abaixo
assets/      Estáticos servidos: CSS/JS compilados, imagens, ícones
files/       PDFs (currículo, históricos)
code/        Notebooks e ambientes por post (fonte reproducível; excluído do build)
scripts/     Pipeline notebook -> post (veja scripts/README.md)
```

Para gerar um post a partir de um notebook em `code/<slug>/`, use o pipeline
documentado em [`scripts/README.md`](scripts/README.md).

## Estilos e assets compilados

`assets/css/main.css` e `assets/js/bundle.js` são **artefatos versionados** — é o que
o site realmente serve. Eles são compilados a partir de `_sass/` e `_js/` pela pipeline
**Gulp** (`gulpfile.js` + `package.json`), que **não roda no fluxo do dia a dia**. Ou seja:
editar `_sass/`/`_js/` sozinho **não** muda o site até recompilar.

Para mexer no visual: instale as dependências de build (`npm install`) e rode o Gulp para
regerar `main.css`/`bundle.js`. **Não edite `main.css` na mão** — ele seria sobrescrito na
próxima compilação.

## Créditos

- Logo / favicon: galinha do [OpenMoji](https://openmoji.org/), **modificada** (recorte do `viewBox` para enquadrar melhor o ícone), licenciada sob [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Como é uma obra derivada sob _ShareAlike_, o ícone (`_includes/logo.svg` e `assets/img/icons/favicon.svg`) permanece sob CC BY-SA 4.0.
