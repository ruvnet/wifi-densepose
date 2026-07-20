// §4.7 Calibration Wizard — baseline → enroll → train → verify.
// Stepped wizard (1–5) against the ADR-151 calibration HTTP API.

import { h, clear, card, pill, statusPill, sectionHeader, bar, banner, button, mono } from '../ui.js';

export default {
  meta: { title: 'Calibration' },
  async render(root, ctx) {
    const { api } = ctx;
    const cal = api.calibration;
    const state = { step: 1, room_id: '', seed: '', baseline_id: null, anchorIdx: 0, trainResult: null };
    // Track the active baseline poll so it can be cancelled on Restart, on a
    // step change, and when the panel itself is torn down (the router only
    // calls the cleanup this render() returns — a per-card _cleanup was never
    // invoked, leaking the setTimeout loop).
    let activePoll = null;
    function stopPoll() {
      if (activePoll) { activePoll.cancelled = true; if (activePoll.timer) clearTimeout(activePoll.timer); activePoll = null; }
    }

    root.appendChild(sectionHeader('Calibration Wizard', 'baseline → enroll → train → verify'));
    if (cal.demo) root.appendChild(banner('DEMO — cog-calibration HTTP API (ADR-151) simulated in-browser; the live service replaces this (§7.1).', 'amber'));
    const stepper = h('.stepper');
    const body = h('div');
    root.appendChild(stepper);
    root.appendChild(body);

    const STEPS = ['Select', 'Baseline', 'Enroll', 'Train', 'Verify'];
    function paintStepper() {
      clear(stepper);
      STEPS.forEach((s, i) => {
        const n = i + 1;
        const cls = n === state.step ? 'active' : (n < state.step ? 'done' : '');
        stepper.appendChild(h('.step-pill' + (cls ? '.' + cls : ''), h('span.n', n < state.step ? '✓' : String(n)), s));
      });
    }
    function go(step) { stopPoll(); state.step = step; paintStepper(); render(); }
    function render() {
      clear(body);
      if (state.step === 1) body.appendChild(step1());
      else if (state.step === 2) body.appendChild(step2());
      else if (state.step === 3) body.appendChild(step3());
      else if (state.step === 4) body.appendChild(step4());
      else body.appendChild(step5());
    }

    // ── Step 1 — select room + SEED ────────────────────────────────
    function step1() {
      const roomInput = h('input.search', { placeholder: 'room_id  (A-Za-z0-9_- , 1–64)', value: state.room_id });
      const seedSel = h('select.inline');
      const warn = h('div');
      let seedList = [];
      (async () => {
        try { seedList = (await api.seeds()).filter((s) => s.online); }
        catch (e) { warn.appendChild(banner('SEED fleet unavailable — ' + (e.message || e), 'red')); }
        seedList.forEach((s) => seedSel.appendChild(h('option', { value: s.device_id }, `${s.device_id} (${s.zone})`)));
      })();
      const validate = () => {
        const ok = /^[A-Za-z0-9_-]{1,64}$/.test(roomInput.value);
        const seed = seedList.find((s) => s.device_id === seedSel.value);
        clear(warn);
        if (!ok) warn.appendChild(banner('room_id must match [A-Za-z0-9_-]{1,64}', 'red'));
        else if (seed && seed.frame_rate_hz < 80) warn.appendChild(banner(`CSI ingest low (${seed.frame_rate_hz} Hz) — a broken pipeline silently fails calibration`, 'amber'));
        return ok;
      };
      roomInput.addEventListener('input', validate);
      seedSel.addEventListener('change', validate);
      return card({
        title: 'Step 1 — Select room and SEED', children: [
          h('h3', 'room_id'), roomInput,
          h('h3.mt', 'Serving SEED'), seedSel, warn,
          h('.mt', button('Next', { variant: 'primary', onClick: () => { if (validate()) { state.room_id = roomInput.value; state.seed = seedSel.value; go(2); } } })),
        ],
      });
    }

    // ── Step 2 — baseline capture ──────────────────────────────────
    function step2() {
      const progress = h('.bar', { style: { height: '14px' } }, h('span'));
      const meta = h('.t2.mt');
      const baselineLine = h('div');
      const c = card({
        title: 'Step 2 — Baseline capture (room must be empty)', children: [
          progress, meta, baselineLine,
          h('.mt', button('Restart', {
            variant: 'ghost',
            // Cancel the in-flight poll loop (was leaked before), reset the
            // session, and start a fresh capture.
            onClick: () => { stopPoll(); cal.reset(); clear(baselineLine); startCapture(); },
          })),
        ],
      });

      // Single-flight: stopPoll() before (re)arming guarantees one loop.
      function startCapture() {
        stopPoll();
        const session = { cancelled: false, timer: null };
        activePoll = session;
        (async () => {
          let startRes;
          try { startRes = await cal.start(); }
          catch (e) { clear(meta); meta.appendChild(banner('Baseline start failed — ' + (e.message || e), 'red')); return; }
          if (session.cancelled) return;
          state.baseline_id = (startRes && startRes.baseline_id) || state.baseline_id;
          const loop = async () => {
            if (session.cancelled) return;
            let st;
            try { st = await cal.status(); }
            catch (e) { clear(meta); meta.appendChild(banner('Status unavailable — ' + (e.message || e), 'red')); return; }
            if (session.cancelled) return;
            progress.firstChild.style.width = pct(st.frames, st.target) + '%';
            clear(meta); meta.appendChild(document.createTextNode(`${st.frames}/${st.target} frames · ETA ${st.eta_s}s · z_median ${st.z_median}`));
            if (st.motion_flagged) { if (!c.querySelector('.banner')) c.insertBefore(banner('Room must be empty — movement detected', 'amber'), progress); }
            else { const b = c.querySelector('.banner'); if (b) b.remove(); }
            if (st.target > 0 && st.frames >= st.target) {
              activePoll = null;
              state.baseline_id = state.baseline_id || 'bl-unknown';
              clear(baselineLine);
              baselineLine.appendChild(h('.mt', h('span.green', 'Baseline complete · '), mono(state.baseline_id), h('span.t2', ' (record this — it anchors STALE detection)')));
              baselineLine.appendChild(h('.mt', button('Continue to enrollment', { variant: 'primary', onClick: () => go(3) })));
              return;
            }
            session.timer = setTimeout(loop, 600);
          };
          loop();
        })();
      }

      startCapture();
      return c;
    }

    // ── Step 3 — anchor enrollment ─────────────────────────────────
    function step3() {
      const anchors = cal.ANCHORS;
      const counter = h('h3', 'enrollment');
      const list = h('div');
      const current = h('div');
      async function paint() {
        let acc;
        try { acc = new Set(((await cal.enrollStatus()).accepted) || []); }
        catch (e) { clear(current); current.appendChild(banner('Enroll status unavailable — ' + (e.message || e), 'red')); acc = new Set(); }
        clear(counter); counter.appendChild(document.createTextNode(`${acc.size} / ${anchors.length} anchors accepted`));
        clear(list);
        anchors.forEach((label, i) => {
          list.appendChild(h('.row', mono(label),
            acc.has(label) ? pill('accepted', 'green') : (i === state.anchorIdx ? pill('current', 'cyan') : pill('pending', 'grey'))));
        });
        clear(current);
        const label = anchors[state.anchorIdx];
        if (!label) {
          current.appendChild(h('.mt', h('span.green', 'All anchors processed · '),
            button('Train specialists', { variant: 'primary', onClick: () => go(4) })));
          return;
        }
        current.appendChild(h('h3.mt', `Anchor: ${label}`));
        current.appendChild(h('.t2', instruction(label)));
        current.appendChild(h('.mt', button('Capture anchor', {
          variant: 'primary', onClick: async () => {
            let r;
            try { r = await cal.anchor(label); }
            catch (e) { current.appendChild(banner('Capture failed — ' + (e.message || e), 'red')); return; }
            const f = r.features;
            const res = h('.mt', r.accepted ? pill('accepted', 'green') : pill('retry', 'amber'),
              r.reason ? h('span.amber', ' ' + r.reason) : null,
              f ? h('.mono.t2.mt', `mean ${f.mean} · var ${f.variance} · breathing ${f.breathing_score} · heart ${f.heart_score}`) : null);
            current.appendChild(res);
            if (r.accepted) { state.anchorIdx++; setTimeout(paint, 700); }
          },
        })));
      }
      paint();
      return card({ title: 'Step 3 — Anchor enrollment', children: [counter, list, current] });
    }

    // ── Step 4 — train ─────────────────────────────────────────────
    function step4() {
      const body4 = h('div', h('.muted-empty', 'Training…'));
      const c = card({ title: 'Step 4 — Train specialists', children: [body4] });
      (async () => {
        let r;
        try { r = await cal.train(state.room_id); }
        catch (e) { clear(body4); body4.appendChild(banner('Training failed — ' + (e.message || e), 'red')); return; }
        state.trainResult = r;
        clear(body4);
        const specs = [
          ['presence', r.presence && `threshold ${r.presence.threshold} · var ${r.presence.occupied_var}`],
          ['posture', r.posture && `${r.posture.prototypes} prototypes`],
          ['breathing', r.breathing && `min_score ${r.breathing.min_score}`],
          ['heartbeat', r.heartbeat && `min_score ${r.heartbeat.min_score}`],
          ['restlessness', r.restlessness && `calm ${r.restlessness.calm} · active ${r.restlessness.active}`],
          ['anomaly', r.anomaly && `${r.anomaly.prototypes} prototypes · scale ${r.anomaly.scale}`],
        ];
        specs.forEach(([name, detail]) => {
          body4.appendChild(h('.row', mono(name),
            detail ? h('.flex.gap-sm', pill('trained', 'green'), h('span.t2', detail))
              : h('.flex.gap-sm', pill('null', 'amber'), button('Re-enroll missing anchors', { variant: 'ghost', onClick: () => go(3) }))));
        });
        body4.appendChild(h('.mt', button('Verify live', { variant: 'primary', onClick: () => go(5) })));
      })();
      return c;
    }

    // ── Step 5 — verify live ───────────────────────────────────────
    function step5() {
      const rows = h('div', h('.muted-empty', 'Loading live RoomState…'));
      (async () => {
        let live;
        try {
          const all = await api.roomStates();
          live = all.find((r) => r.room_id === state.room_id) || all[0];
        } catch (e) { clear(rows); rows.appendChild(banner('Live RoomState unavailable — ' + (e.message || e), 'red')); return; }
        clear(rows);
        if (!live) { rows.appendChild(h('.muted-empty', 'No RoomState yet — give the room a moment after training.')); return; }
        rows.appendChild(h('.row', 'Presence', live.presence ? statusPill(live.presence.value) : h('span.t3', '—')));
        rows.appendChild(h('.row', 'Posture', live.posture ? statusPill(live.posture.value) : h('span.t3', '—')));
        rows.appendChild(h('.row', 'Breathing', h('span.cyan', live.breathing_bpm ? live.breathing_bpm.value + ' BPM' : '—')));
        rows.appendChild(h('.row', 'Heart rate', h('span.cyan', live.heart_bpm ? live.heart_bpm.value + ' BPM' : '—')));
      })();
      return card({
        title: 'Step 5 — Verify live', children: [
          h('.t2', 'Stand in the room to confirm presence; sit/lie to confirm posture; breathe normally to confirm vitals.'),
          rows,
          h('.flex.mt',
            button('Confirm and save', { variant: 'primary', onClick: () => { cal.reset && cal.reset(); ctx.navigate('#/rooms'); } }),
            button("Something's wrong — re-enroll", { variant: 'ghost', onClick: () => go(3) })),
        ],
      });
    }

    paintStepper();
    render();
    // The router invokes this on navigation away — tear down any live poll.
    return () => stopPoll();
  },
};

// Guard against NaN%/Infinity% when target is 0/missing (§4.7 robustness).
function pct(frames, target) {
  if (!(target > 0)) return 0;
  return Math.max(0, Math.min(100, (frames / target) * 100)).toFixed(0);
}

function instruction(label) {
  const map = {
    empty: 'Leave the room empty and still.',
    stand_still: 'Stand still in the centre of the room.',
    sit: 'Sit down naturally.',
    lie_down: 'Lie down (bed/sofa).',
    breathe_slow: 'Breathe slowly and deeply.',
    breathe_normal: 'Breathe at your normal resting rate.',
    small_move: 'Make small fidgeting movements.',
    sleep_posture: 'Adopt your typical sleeping posture and stay still.',
  };
  return map[label] || label;
}
