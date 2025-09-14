(function () {
  const qs = (s, r = document) => r.querySelector(s);
  const elBtn    = qs('#wx-toggle');
  const elBadge  = qs('#wx-badge');
  const elEmoji  = qs('#wx-emoji');
  const elPanel  = qs('#wx-panel');
  const elDays   = qs('#wx-days');
  const elNow    = qs('#wx-now-desc');

  if (!elBtn || !elBadge || !elPanel) return;

  let open = false;
  elBtn.addEventListener('click', () => {
    open = !open;
    elBtn.setAttribute('aria-expanded', String(open));
    elPanel.hidden = !open;
  });

  const DEFAULT_LAT = 40.4168; // Madrid
  const DEFAULT_LON = -3.7038;

  async function loadWeather(lat = DEFAULT_LAT, lon = DEFAULT_LON, days = 8) {
    const u = new URL('/api/weather', location.origin);
    u.searchParams.set('lat', lat);
    u.searchParams.set('lon', lon);
    u.searchParams.set('days', days);

    try {
      const res = await fetch(u, { headers: { 'Accept': 'application/json' } });
      const json = await res.json();
      if (!json.ok) throw new Error(json.error || 'Weather error');

      const { current, daily } = json.data || {};
      // Actual
      if (current && typeof current.temp === 'number') elBadge.textContent = Math.round(current.temp) + '°';
      if (current && current.emoji) elEmoji.textContent = current.emoji;
      if (current && current.desc)  elNow.textContent   = current.desc;

      // Días
      elDays.innerHTML = '';
      (daily || []).forEach(d => {
        const li = document.createElement('li');
        const date = new Date(d.date + 'T00:00:00');
        const fmt = date.toLocaleDateString('es-ES', { weekday: 'short', day: '2-digit', month: '2-digit' });
        li.innerHTML = `
          <span>${fmt} ${d.emoji || ''}</span>
          <span title="Mínima">${Math.round(d.tmin)}°</span>
          <strong title="Máxima">${Math.round(d.tmax)}°</strong>
        `;
        elDays.appendChild(li);
      });
    } catch (err) {
      console.error('Weather widget:', err);
      elBadge.textContent = '--°';
      elNow.textContent = 'Sin datos';
    }
  }

  // Carga inicial (Madrid)
  loadWeather();
})();