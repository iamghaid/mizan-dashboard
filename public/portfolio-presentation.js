(() => {
  const config = {"languageKey": "mizan_lang", "themeKey": "mizan_theme"};
  const callbacks = new Set();
  const preferences = {};
  const validLanguage = v => v === 'en' || v === 'ar';
  const validTheme = v => v === 'light' || v === 'dark';
  function apply(data, notify) {
    if (validLanguage(data.language)) {
      preferences.language = data.language;
      document.documentElement.lang = data.language;
      document.documentElement.dir = data.language === 'ar' ? 'rtl' : 'ltr';
      try { localStorage.setItem(config.languageKey, config.jsonLanguage ? JSON.stringify(data.language) : data.language); } catch {}
    }
    if (validTheme(data.theme)) {
      preferences.theme = data.theme;
      document.documentElement.dataset.theme = data.theme;
      try { localStorage.setItem(config.themeKey, data.theme); } catch {}
    }
    if (notify) callbacks.forEach(callback => callback({ ...preferences }));
  }
  const query = new URLSearchParams(location.search);
  apply({ language: query.get('portfolio_lang'), theme: query.get('portfolio_theme') }, false);
  window.PortfolioPresentation = {
    get language() { return preferences.language; },
    get theme() { return preferences.theme; },
    subscribe(callback) { callbacks.add(callback); return () => callbacks.delete(callback); }
  };
  window.addEventListener('message', event => {
    if (event.source !== window.parent || event.origin !== 'https://gheid-mycv.vercel.app') return;
    if (!event.data || event.data.type !== 'portfolio:presentation') return;
    apply(event.data, true);
  });
})();
