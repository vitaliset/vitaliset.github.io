---
layout: page
title: Gallery
permalink: /gallery/
sitemap: false
---

<style>
  .gallery-hero { margin: 0 0 1.5rem; }
  .gallery-hero img { width: 100%; border-radius: 8px; display: block; }
  .gallery-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1.25rem;
    margin: 2rem 0;
  }
  .gallery-grid figure { margin: 0; }
  .gallery-grid img {
    width: 100%;
    aspect-ratio: 16 / 9;
    object-fit: cover;
    border-radius: 6px;
    display: block;
    transition: transform .2s ease;
  }
  .gallery-grid a:hover img { transform: scale(1.03); }
  .gallery-grid figcaption {
    font-size: .85rem;
    margin-top: .5rem;
    text-align: center;
    opacity: .8;
  }
</style>

<div class="gallery-hero">
  <img src="{{ site.baseurl }}/assets/img/desenhos/teste1.jpg" alt="O robô-mascote do blog, ilustração de Lucas Álamo">
</div>

<p><div align="justify">As ilustrações abaixo são a arte original que deu identidade ao <i>Vitali Set</i> por anos, feitas à mão pelo talentoso <a href="https://www.instagram.com/lucasalamoart/">Lucas Álamo</a> — cada capa uma metáfora visual do tema do post. Esta página é uma homenagem e um acervo desse trabalho. <i>(These are the original hand-made illustrations by <a href="https://www.instagram.com/lucasalamoart/">Lucas Álamo</a> that defined the blog's visual identity — kept here as a tribute and archive.)</i></div></p>

<div class="gallery-grid">

  <figure>
    <a href="{{ site.baseurl }}/distance_metrics/">
      <img loading="lazy" src="{{ site.baseurl }}/assets/img/posts/coverdistancia_md.jpg" alt="Generalizando distância">
    </a>
    <figcaption>Generalizando distância</figcaption>
  </figure>

  <figure>
    <a href="{{ site.baseurl }}/k-fold/">
      <img loading="lazy" src="{{ site.baseurl }}/assets/img/posts/coverkfold_md.jpg" alt="Motivando k-Fold">
    </a>
    <figcaption>Motivando k-Fold</figcaption>
  </figure>

  <figure>
    <a href="{{ site.baseurl }}/covariate-shift-0-introduction/">
      <img loading="lazy" src="{{ site.baseurl }}/assets/img/posts/covariate_0_formulando_md.jpg" alt="Covariate Shift: Introduction">
    </a>
    <figcaption>Covariate Shift: Introduction</figcaption>
  </figure>

  <figure>
    <a href="{{ site.baseurl }}/covariate-shift-1-qqplot/">
      <img loading="lazy" src="{{ site.baseurl }}/assets/img/posts/coverqqplot_md.jpg" alt="Covariate Shift: QQ-plot">
    </a>
    <figcaption>Covariate Shift: QQ-plot</figcaption>
  </figure>

  <figure>
    <a href="{{ site.baseurl }}/covariate-shift-2-classificador-binario/">
      <img loading="lazy" src="{{ site.baseurl }}/assets/img/posts/coverclassificador_binario_md.jpg" alt="Covariate Shift: Classificador Binário">
    </a>
    <figcaption>Covariate Shift: Classificador Binário</figcaption>
  </figure>

  <figure>
    <a href="{{ site.baseurl }}/boruta/">
      <img loading="lazy" src="{{ site.baseurl }}/assets/img/posts/boruta_md.jpg" alt="Uma utilização crítica do Boruta">
    </a>
    <figcaption>Uma utilização crítica do Boruta</figcaption>
  </figure>

  <figure>
    <a href="{{ site.baseurl }}/metakmeans/">
      <img loading="lazy" src="{{ site.baseurl }}/assets/img/posts/metakmeans_md.jpg" alt="Meta K-Means: um ensemble de K-Means">
    </a>
    <figcaption>Meta K-Means: um ensemble de K-Means</figcaption>
  </figure>

  <figure>
    <a href="{{ site.baseurl }}/threshold-dependent-opt/">
      <img loading="lazy" src="{{ site.baseurl }}/assets/img/posts/threshold_dependent_opt_md.jpg" alt="Hyperparameter search with threshold-dependent metrics">
    </a>
    <figcaption>Hyperparameter search with threshold-dependent metrics</figcaption>
  </figure>

  <figure>
    <a href="{{ site.baseurl }}/conditional-density-estimation/">
      <img loading="lazy" src="{{ site.baseurl }}/assets/img/posts/cde_md.jpg" alt="Conditional Density Estimation (esboço)">
    </a>
    <figcaption>Conditional Density Estimation <i>(esboço / sketch)</i></figcaption>
  </figure>

  <figure>
    <img loading="lazy" src="{{ site.baseurl }}/assets/img/desenhos/teste2.jpg" alt="O robô-mascote e seu gato, ilustração de Lucas Álamo">
    <figcaption>O mascote &amp; cia. <i>(mascote do site)</i></figcaption>
  </figure>

</div>
