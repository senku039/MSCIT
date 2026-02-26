(function () {
  function isMultiline(el) {
    return !!el && (el.tagName === 'TEXTAREA' || el.isContentEditable);
  }

  function findPrimaryAction(scope) {
    if (!scope) return null;
    return scope.querySelector('[data-primary-action]:not([disabled]), button[type="submit"]:not([disabled]), .btn-primary:not([disabled])');
  }

  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter') return;
    const target = event.target;
    if (isMultiline(target)) return;
    if (target && target.closest('[role="dialog"]')) {
      const dialog = target.closest('[role="dialog"]');
      const primary = findPrimaryAction(dialog);
      if (primary && primary !== target) {
        event.preventDefault();
        primary.click();
      }
      return;
    }

    const form = target && target.closest('form');
    if (form) {
      const submit = findPrimaryAction(form);
      if (submit && submit !== target) {
        event.preventDefault();
        submit.click();
      }
    }
  });

  let trappedModal = null;
  let previousFocus = null;

  function keyHandler(e) {
    if (e.key !== 'Tab' || !trappedModal) return;
    const focusables = trappedModal.querySelectorAll('a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])');
    if (!focusables.length) return;
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }

  window.AppUI = {
    activateFocusTrap(modal) {
      trappedModal = modal;
      previousFocus = document.activeElement;
      document.addEventListener('keydown', keyHandler);
    },
    deactivateFocusTrap() {
      document.removeEventListener('keydown', keyHandler);
      trappedModal = null;
      if (previousFocus && typeof previousFocus.focus === 'function') previousFocus.focus();
      previousFocus = null;
    }
  };
})();
