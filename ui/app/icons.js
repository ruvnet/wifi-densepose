// Inline SVG icons (stroke=currentColor). Kept minimal — 24x24 line icons.
const s = (paths) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths}</svg>`;

export const icons = {
  dashboard: s('<rect x="3" y="3" width="7" height="9" rx="1.5"/><rect x="14" y="3" width="7" height="5" rx="1.5"/><rect x="14" y="12" width="7" height="9" rx="1.5"/><rect x="3" y="16" width="7" height="5" rx="1.5"/>'),
  sensing: s('<path d="M12 12a0 0 0 100 0"/><circle cx="12" cy="12" r="1.5"/><path d="M8.5 8.5a5 5 0 000 7M15.5 8.5a5 5 0 010 7M5.6 5.6a9 9 0 000 12.8M18.4 5.6a9 9 0 010 12.8"/>'),
  nodes: s('<rect x="3" y="3" width="8" height="8" rx="1.5"/><rect x="13" y="3" width="8" height="8" rx="1.5"/><rect x="3" y="13" width="8" height="8" rx="1.5"/><rect x="13" y="13" width="8" height="8" rx="1.5"/>'),
  demo: s('<polygon points="6 4 20 12 6 20 6 4"/>'),
  training: s('<path d="M12 3v4M5 7l2.5 2.5M19 7l-2.5 2.5"/><circle cx="12" cy="14" r="6"/><path d="M12 11v3l2 2"/>'),
  about: s('<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 7.5h.01"/>'),
  fusion: s('<circle cx="8" cy="12" r="4"/><circle cx="16" cy="12" r="4"/>'),
  observatory: s('<path d="M3 20h18M5 20l5-9 4 5 5-9"/><circle cx="19" cy="7" r="2"/>'),
  ext: s('<path d="M14 4h6v6M20 4l-9 9M9 5H5a1 1 0 00-1 1v13a1 1 0 001 1h13a1 1 0 001-1v-4"/>'),
  logo: s('<circle cx="12" cy="12" r="2"/><path d="M7 7a7 7 0 000 10M17 7a7 7 0 010 10"/>'),
};
