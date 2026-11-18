(function () {
  const content = document.querySelector('article.post-content, .page-content');
  if (!content || !window.katex || !window.renderMathInElement) return;

  function formatMath(source) {
    const condition = '\\left(Y \\,|\\, X_1=x_1, X_2=x_2,\\cdots, X_n=x_n\\right)';
    const breakableCondition = '\\bigl(Y \\,|\\,\\allowbreak {X_1=x_1, X_2=x_2,}\\,\\allowbreak {\\cdots, X_n=x_n}\\bigr)';
    const loss = '\\left(\\left(h(X) - Y\\right)^2\\right)';
    return source.trim()
      .replace(/^\\begin\{equation\*\}([\s\S]*?)\\end\{equation\*\}$/, '$1')
      .replace('\\,\\mathbb{E}_{', '\\,\\allowbreak\\mathbb{E}_{')
      .replace(' ' + loss, ' \\allowbreak' + loss)
      .replace(condition, breakableCondition);
  }

  const options = { preProcess: formatMath, strict: 'ignore', throwOnError: false };
  const blocks = ['equation', 'equation*', 'align', 'align*', 'alignat', 'alignat*', 'gather', 'gather*'];
  renderMathInElement(content, {
    ...options,
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '\\[', right: '\\]', display: true },
      ...blocks.map(name => ({ left: '\\begin{' + name + '}', right: '\\end{' + name + '}', display: true })),
      { left: '\\(', right: '\\)', display: false },
      { left: '$', right: '$', display: false }
    ],
    ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option']
  });

  // Support the script elements produced by older Kramdown builds.
  for (const script of content.querySelectorAll('script[type^="math/tex"]')) {
    const formula = document.createElement('span');
    katex.render(formatMath(script.textContent), formula, {
      strict: options.strict,
      throwOnError: options.throwOnError,
      displayMode: script.type.includes('mode=display')
    });
    script.replaceWith(formula);
  }

  const formulas = Array.from(content.querySelectorAll('.katex')).map(element => ({
    element,
    container: element.closest('.katex-display, p, div, li, th, td, blockquote, article'),
    parts: element.querySelectorAll('.katex-html > .katex-base, .katex-html > .katex-tag')
  }));
  let frame;

  function fitMath() {
    // KaTeX can wrap between bases. Fit only when an individual piece is too wide.
    for (const { element } of formulas) element.style.removeProperty('--math-fit');
    const scales = formulas.map(({ container, parts }) => {
      if (!container || !container.clientWidth) return 1;
      const { paddingLeft, paddingRight } = getComputedStyle(container);
      const width = container.clientWidth - parseFloat(paddingLeft) - parseFloat(paddingRight) - 2;
      if (width <= 0) return 1;
      const widest = Math.max(0, ...Array.from(parts, part => part.getBoundingClientRect().width));
      return widest > width ? width / widest : 1;
    });
    formulas.forEach(({ element }, index) => {
      if (scales[index] < 1) element.style.setProperty('--math-fit', scales[index]);
    });
  }

  function scheduleFit() {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(fitMath);
  }

  fitMath();
  if (window.ResizeObserver) {
    let lastWidth = content.clientWidth;
    new ResizeObserver(() => {
      const width = content.clientWidth;
      if (width !== lastWidth) {
        lastWidth = width;
        scheduleFit();
      }
    }).observe(content);
  } else {
    window.addEventListener('resize', scheduleFit);
  }
  new MutationObserver(scheduleFit).observe(document.documentElement, {
    attributes: true, attributeFilter: ['data-lang']
  });
  (document.fonts ? document.fonts.ready : Promise.resolve()).then(() => {
    fitMath();
    document.dispatchEvent(new Event('blog:math-ready'));
  });
})();
