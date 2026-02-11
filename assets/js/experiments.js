(() => {
    const blocks = document.querySelectorAll('[data-exp]');
    if (!blocks.length) return;
  
    function setActive(block, idx) {
      const pages = block.querySelectorAll('[data-exp-page]');
      const tabs  = block.querySelectorAll('[data-exp-tab]');
      const dots  = block.querySelectorAll('[data-exp-dot]');
  
      const n = pages.length;
      const k = ((idx % n) + n) % n;
  
      pages.forEach(p => p.classList.toggle('is-active', Number(p.dataset.expPage) === k));
      tabs.forEach(t => {
        const on = Number(t.dataset.expTab) === k;
        t.classList.toggle('is-active', on);
        t.setAttribute('aria-selected', on ? 'true' : 'false');
      });
      dots.forEach(d => d.classList.toggle('is-active', Number(d.dataset.expDot) === k));
  
      // If MathJax is present, re-typeset the newly shown page
      if (window.MathJax?.typeset) window.MathJax.typeset();
    }
  
    blocks.forEach(block => {
      const pages = block.querySelectorAll('[data-exp-page]');
      if (!pages.length) return;
  
      let idx = 0;
  
      block.querySelector('[data-exp-prev]')?.addEventListener('click', () => {
        idx -= 1; setActive(block, idx);
      });
  
      block.querySelector('[data-exp-next]')?.addEventListener('click', () => {
        idx += 1; setActive(block, idx);
      });
  
      block.querySelectorAll('[data-exp-tab]').forEach(btn => {
        btn.addEventListener('click', () => {
          idx = Number(btn.dataset.expTab);
          setActive(block, idx);
        });
      });
  
      block.querySelectorAll('[data-exp-dot]').forEach(btn => {
        btn.addEventListener('click', () => {
          idx = Number(btn.dataset.expDot);
          setActive(block, idx);
        });
      });
  
      // init
      setActive(block, 0);
    });
  })();