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
