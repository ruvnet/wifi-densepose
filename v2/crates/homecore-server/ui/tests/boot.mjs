// Boot regression test — exercises the REAL app.js boot + router (not
// just individual panels). Catches the class of bug where start() throws
// before route() runs and the dashboard renders blank.
// Run: node tests/boot.mjs   (from the ui/ dir)
import { install } from './dom-shim.mjs';
const { document, window } = install();
globalThis.HOMECORE_UI_DEMO = true; // boot with fixtures (no gateway in tests)

const errs = [];
const origErr = console.error;
console.error = (...a) => { errs.push(a.map(String).join(' ')); };

await import('../js/app.js');
await new Promise((r) => setTimeout(r, 30));
console.error = origErr;

const fails = [];
const content = document.getElementById('hc-content');
const app = document.getElementById('app');

if (!app || app.children.length < 2) fails.push('shell not built (#app should have topnav + shell)');
if (!content) fails.push('#hc-content missing — buildShell did not run');
else if (content.children.length === 0) fails.push('BLANK: dashboard rendered nothing into #hc-content on boot');
if (errs.length) fails.push('console.error during boot: ' + errs.slice(0, 3).join(' | '));

// navigation must re-render the panel
window.location.hash = '#/fleet';
await new Promise((r) => setTimeout(r, 30));
if (!content || content.children.length === 0) fails.push('BLANK after navigating to #/fleet');

// a clean topnav with no dead Cognitum tabs / Cog Store link
const links = app ? app.querySelectorAll('a') : [];
const hrefs = links.map((a) => a.getAttribute('href') || '');
if (hrefs.some((h) => /cognitum\.one\/store/.test(h))) fails.push('Cog Store external link should be removed');

if (fails.length) { console.error('\nFAILED:'); fails.forEach((f) => console.error('  ✗ ' + f)); process.exit(1); }
console.log('OK — app.js boots, dashboard renders, navigation re-renders, no dead Cog Store link');
