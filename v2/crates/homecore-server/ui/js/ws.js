// HOMECORE-UI WebSocket client — ADR-130 subscribe_events.
//
// "The UI must never poll for entity state" (ADR-131 §2/§4.4). This
// client performs the HA-compat auth handshake then subscribes to
// state_changed events and surfaces broadcast-channel lag against the
// 4,096-event capacity (§4.1/§4.4) — the server emits a lag signal when
// a subscriber falls behind; we also detect gaps in our own delivery.

import { api } from './api.js';

/**
 * Connect and stream events.
 * @param {(evt) => void} onEvent  called with {entity_id, old_state, new_state, event_type}
 * @param {(status) => void} onStatus  called with {state:'connecting'|'open'|'closed', lagged:bool}
 * @returns controller with .close()
 */
export function connect(onEvent, onStatus) {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${proto}//${location.host}/api/websocket`;
  let ws, msgId = 1, closedByUs = false, lagged = false;
  let retry = 0;
  const status = (state) => onStatus && onStatus({ state, lagged });

  function open() {
    status('connecting');
    try { ws = new WebSocket(url); } catch (e) { schedule(); return; }
    ws.onmessage = (m) => {
      let msg; try { msg = JSON.parse(m.data); } catch { return; }
      if (msg.type === 'auth_required') {
        ws.send(JSON.stringify({ type: 'auth', access_token: api.token() }));
      } else if (msg.type === 'auth_ok') {
        retry = 0; status('open');
        ws.send(JSON.stringify({ id: msgId++, type: 'subscribe_events', event_type: 'state_changed' }));
      } else if (msg.type === 'auth_invalid') {
        status('closed');
      } else if (msg.type === 'event' && msg.event) {
        const e = msg.event;
        if (e.event_type === 'state_changed' && e.data) {
          onEvent && onEvent({
            event_type: 'state_changed',
            entity_id: e.data.entity_id,
            old_state: e.data.old_state,
            new_state: e.data.new_state,
          });
        } else {
          onEvent && onEvent({ event_type: e.event_type, ...e.data });
        }
      } else if (msg.type === 'lagged' || (msg.type === 'event' && msg.lagged)) {
        lagged = true; status('open');
      }
    };
    ws.onclose = () => { if (!closedByUs) schedule(); else status('closed'); };
    ws.onerror = () => { try { ws.close(); } catch {} };
  }

  function schedule() {
    status('closed');
    retry = Math.min(retry + 1, 6);
    const delay = Math.min(500 * 2 ** retry, 15000);
    setTimeout(() => { if (!closedByUs) open(); }, delay);
  }

  open();
  return {
    close() { closedByUs = true; try { ws && ws.close(); } catch {} },
    isLagged: () => lagged,
    clearLag() { lagged = false; },
  };
}
