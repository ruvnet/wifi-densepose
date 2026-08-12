const WS_PORT_BY_HTTP_PORT = {
  // Standalone development UI and sensing server.
  '8080': '8765',
};

/**
 * Derive the sensing WebSocket exposed alongside the current HTTP mapping.
 * Docker documents adjacent host ports (HTTP N, sensing N+1); deployments
 * without an explicit page port are expected to reverse-proxy both paths on
 * the same origin.
 */
export function buildPoseFusionWsUrl(
  locationLike = (typeof window !== 'undefined' ? window.location : null),
) {
  const protocol = locationLike?.protocol === 'https:' ? 'wss:' : 'ws:';
  const hostname = locationLike?.hostname || 'localhost';
  const pagePort = locationLike?.port || '';
  let wsPort = WS_PORT_BY_HTTP_PORT[pagePort] || '';

  if (!wsPort && /^\d+$/.test(pagePort)) {
    const numericPort = Number(pagePort);
    if (numericPort > 0 && numericPort < 65535) {
      wsPort = String(numericPort + 1);
    }
  }

  const host = wsPort ? `${hostname}:${wsPort}` : (locationLike?.host || hostname);
  return `${protocol}//${host}/ws/sensing`;
}

const SOURCE_STATES = {
  connecting: { label: 'CONNECTING CSI', dotClass: 'offline', button: 'Connecting...' },
  live: { label: 'LIVE CSI', dotClass: null, button: '✓ Connected' },
  'server-simulated': {
    label: 'SERVER-SIMULATED CSI',
    dotClass: 'warning',
    button: 'Server Simulation',
  },
  offline: { label: 'CSI OFFLINE / STALE', dotClass: 'warning', button: 'Reconnect' },
  unverified: {
    label: 'CONNECTED CSI — SOURCE UNVERIFIED',
    dotClass: 'warning',
    button: 'Connected',
  },
  simulated: { label: 'SIMULATED CSI', dotClass: 'warning', button: 'Connect' },
};

/** Render an explicit CSI provenance state at the page's status boundary. */
export function renderPoseFusionSourceState(source, elements) {
  const state = SOURCE_STATES[source] || SOURCE_STATES.simulated;
  const { statusDot, statusLabel, connectButton } = elements;

  if (statusLabel) statusLabel.textContent = state.label;
  if (statusDot) {
    statusDot.classList.remove('offline', 'warning');
    if (state.dotClass) statusDot.classList.add(state.dotClass);
  }
  if (connectButton) {
    connectButton.textContent = state.button;
    connectButton.classList.remove('active');
    if (source === 'live') connectButton.classList.add('active');
  }
}
