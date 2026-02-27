(function () {
  function isVisible(el) {
    if (!el) return false;
    const style = window.getComputedStyle(el);
    return style.display !== 'none' && style.visibility !== 'hidden' && !el.disabled;
  }

  function pickAction(scope) {
    const selectors = [
      'button[data-enter-action]',
      'button[type="submit"]',
      '#submitBtn',
      '#submitAnswerBtn',
      '#startBtn',
      '#stopBtn',
      '#finishTestBtn',
      '#loginBtn'
    ];
    for (const sel of selectors) {
      const btn = scope.querySelector(sel);
      if (isVisible(btn)) return btn;
    }
    return null;
  }

  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter') return;
    const t = event.target;
    if (!t) return;
    if (t.tagName === 'TEXTAREA' || t.isContentEditable) return;

    const form = t.closest('form');
    const scope = form || document;
    const action = pickAction(scope);
    if (action && action !== t) {
      event.preventDefault();
      action.click();
    }
  });
})();
