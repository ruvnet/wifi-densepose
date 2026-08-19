// About — the former Architecture / Performance / Applications / hero content,
// collapsed into one low-priority reference page so it stays out of the tools.
import { icons } from '../icons.js';
import { html } from '../lib.js';

const PIPELINE = [
  ['CSI input', 'Channel State Information from the WiFi antenna array'],
  ['Phase sanitization', 'Remove hardware-specific noise, normalize signal phase'],
  ['Feature extraction', 'Variance, motion/breath bands, spectral power, change points'],
  ['Inference', 'Presence, posture, vital signs and (optional) pose keypoints'],
  ['Fusion & output', 'Multistatic fusion across nodes → trusted sensing output'],
];
const APPS = [
  ['🛟', 'Elderly & fall monitoring', 'Detect falls and routine anomalies without cameras.'],
  ['🏠', 'Presence & security', 'Through-wall occupancy and intrusion sensing, no line of sight.'],
  ['🏥', 'Patient monitoring', 'Breathing and heart-rate trends from CSI, contact-free.'],
  ['🏢', 'Smart buildings', 'Occupancy-driven HVAC / lighting without privacy intrusion.'],
];

export default {
  id: 'about',
  label: 'About',
  icon: icons.about,

  async mount(root) {
    root.appendChild(html`
      <section class="space-y-6">
        <div class="card card-pad space-y-2">
          <h2 class="text-xl font-bold">RuView — camera-free WiFi sensing</h2>
          <p class="text-ink-soft">Presence, vital-sign and pose estimation from ordinary WiFi Channel State Information. No cameras, no microphones — just RF signal analysis, running on low-cost ESP32 nodes.</p>
          <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
            ${[['🏠', 'Through walls'], ['🔒', 'Privacy-preserving'], ['⚡', 'Real-time'], ['💰', 'Low-cost HW']]
              .map(([i, t]) => `<div class="stat items-center text-center"><span class="text-2xl">${i}</span><span class="text-xs text-ink-muted">${t}</span></div>`).join('')}
          </div>
        </div>

        <div class="card card-pad space-y-3">
          <h3 class="card-title">Processing pipeline</h3>
          <ol class="space-y-2">
            ${PIPELINE.map(([t, d], i) => `<li class="flex gap-3">
              <span class="shrink-0 w-6 h-6 rounded-full bg-brand-500/20 text-brand-300 text-xs font-bold grid place-items-center">${i + 1}</span>
              <div><div class="font-medium text-sm">${t}</div><div class="text-xs text-ink-muted">${d}</div></div></li>`).join('')}
          </ol>
        </div>

        <div class="card card-pad space-y-3">
          <h3 class="card-title">Reference performance</h3>
          <p class="text-xs text-ink-muted">Published WiFi-DensePose benchmark (same-layout). Live accuracy depends on calibration and hardware.</p>
          <div class="grid grid-cols-3 gap-3">
            ${[['AP@50', '87.2%'], ['Avg precision', '43.5%'], ['AP@75', '44.6%']]
              .map(([l, v]) => `<div class="stat"><span class="stat-value text-brand-300">${v}</span><span class="stat-label">${l}</span></div>`).join('')}
          </div>
        </div>

        <div class="card card-pad space-y-3">
          <h3 class="card-title">Applications</h3>
          <div class="grid gap-3 sm:grid-cols-2">
            ${APPS.map(([i, t, d]) => `<div class="rounded-lg bg-ink-1 border border-ink-3 p-3">
              <div class="text-xl mb-1">${i}</div><div class="font-medium text-sm">${t}</div><div class="text-xs text-ink-muted">${d}</div></div>`).join('')}
          </div>
        </div>

        <div class="card card-pad flex flex-wrap gap-3">
          <a href="pose-fusion.html" class="btn-ghost">Pose Fusion visualizer</a>
          <a href="observatory.html" class="btn-ghost">Observatory</a>
          <a href="/api/v1/info" class="btn-ghost">API info</a>
        </div>
        <p class="text-center text-xs text-ink-muted pb-2">RuView · WiFi-DensePose v2</p>
      </section>`);
  },
};
