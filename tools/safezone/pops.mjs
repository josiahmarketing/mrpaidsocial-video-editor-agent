// pops.mjs <index.html> [--fps 30] [--skip sel,sel] [--gap 20] [--jump 100] — find elements that are SHOWN BEFORE THEIR
// ENTRANCE in a HyperFrames / GSAP composition. Steps the timeline one frame at a time and reports two signatures:
//   POP   an element is on screen, then in ONE frame vanishes or collapses (opacity x size falls below 30 %), and
//         within <gap> frames is back. This is what a delayed `fromTo(..., { immediateRender: false })` does to an
//         element with no hidden initial state: it sits in its final place (often inside a parent that is fading
//         in), snaps to the tween's start, and then makes its entrance. An ordinary fade-out is not reported.
//   JUMP  an element at rest moves more than <jump> px in one frame while visible, then keeps moving: it was parked at
//         its final position before a slide-in that starts somewhere else.
// Neither shows on a sheet with one frame every half second, and `hyperframes check` passes both. Fix: give the element
// the tween's start state up front (a CSS class, or gsap.set at load), then run this again.
// "Visible" = not display:none / visibility:hidden, opacity (multiplied up the tree) above 0.08, and a box of 3 px or
// more each way inside the frame. A clip-path that hides an element is not seen: treat wipes by hand.
// Skip things that blink or are moved on purpose (a text cursor, a video layer repositioned while its frames are blank).
import { createRequire } from "node:module"; import { resolve } from "node:path";
const require = createRequire(import.meta.url); const { chromium } = require("playwright");
const args = process.argv.slice(2); const file = resolve(args[0]); const opt = (k, d) => { const i = args.indexOf(k); return i > 0 ? args[i + 1] : d; };
const fps = +opt("--fps", 30), gap = +opt("--gap", 20), jump = +opt("--jump", 100), skip = opt("--skip", "");
const br = await chromium.launch(); const pg = await br.newPage({ viewport: { width: 1080, height: 1920 } }); await pg.goto("file://" + file);
await pg.waitForFunction(() => window.__timelines && Object.keys(window.__timelines).length); await pg.evaluate(() => document.fonts.ready);
const dur = await pg.evaluate(() => Object.values(window.__timelines)[0].duration());
await pg.evaluate((skip) => { let n = 0; window.__els = [];
  for (const e of document.querySelectorAll("#stage *")) { if (["SCRIPT", "STYLE", "AUDIO"].includes(e.tagName) || e.closest("svg") && e.tagName !== "svg" || (skip && e.closest(skip))) continue;
    e.__k = e.id ? "#" + e.id : (e.closest("[id]") ? "#" + e.closest("[id]").id : "") + ">" + e.tagName.toLowerCase() + (e.className && typeof e.className === "string" ? "." + e.className.trim().split(/\s+/).join(".") : "") + "~" + n; n++; window.__els.push(e); } }, skip);
const sample = (t) => pg.evaluate((t) => { Object.values(window.__timelines)[0].seek(t, false); const op = new Map();
  const eff = (e) => { if (!e || e.id === "stage") return 1; if (op.has(e)) return op.get(e); const s = getComputedStyle(e); const v = (s.display === "none" || s.visibility === "hidden") ? 0 : +s.opacity * eff(e.parentElement); op.set(e, v); return v; };
  return window.__els.map((e) => { const o = eff(e); if (o <= 0.08) return null; const q = e.getBoundingClientRect(); if (q.width < 3 || q.height < 3 || q.right <= 0 || q.left >= 1080 || q.bottom <= 0 || q.top >= 1920) return null; return [Math.round(q.left + q.width / 2), Math.round(q.top + q.height / 2), +o.toFixed(2), Math.sqrt(q.width * q.height)]; }); }, t);
const keys = await pg.evaluate(() => window.__els.map((e) => e.__k)); const par = await pg.evaluate(() => window.__els.map((e) => { for (let n = e.parentElement; n; n = n.parentElement) { const i = window.__els.indexOf(n); if (i >= 0) return i; } return -1; })); const N = Math.floor(dur * fps);
const last = keys.map(() => null), still = keys.map(() => 0), hist = keys.map(() => [0, 0, 0]), drop = keys.map(() => null); const out = [];
for (let f = 0; f <= N; f++) { const s = await sample(f / fps);
  for (let i = 0; i < keys.length; i++) { const c = s[i], p = last[i], h = hist[i]; const v = c ? c[2] * c[3] : 0, pv = h[2], mx = Math.max(h[0], h[1], h[2]);
    if (drop[i] && f - drop[i].f > gap) drop[i] = null;
    if (drop[i] && v >= 0.85 * drop[i].v) { out.push({ kind: "POP", i, k: keys[i], t: drop[i].f / fps, n: f - drop[i].f, gone: drop[i].gone }); drop[i] = null; }
    else if (!drop[i] && pv > 0 && pv >= 0.6 * mx && v < 0.3 * pv) drop[i] = { f, v: pv, gone: v === 0 };   // from rest to (nearly) nothing in one frame
    if (c && p) { const d = Math.hypot(c[0] - p[0], c[1] - p[1]); if (d > jump && still[i] >= 1 && p[2] > 0.2 && c[2] > 0.2) out.push({ kind: "JUMP", i, k: keys[i], t: f / fps, n: Math.round(d) }); still[i] = d < 1.5 ? still[i] + 1 : 0; } else still[i] = 0;
    h.shift(); h.push(v); last[i] = c; } }
await br.close();
// a parent that pops takes its children with it: report the outermost element of each event only
const anc = (i) => { const a = []; for (let n = par[i]; n >= 0; n = par[n]) a.push(n); return a; };
const top = out.filter((a) => !anc(a.i).some((n) => out.some((b) => b.i === n && b.kind === a.kind && Math.abs(b.t - a.t) < 1.5 / fps)));
console.log(`${keys.length} elements, ${N + 1} frames: ${top.filter((x) => x.kind === "POP").length} shown before their entrance, ${top.filter((x) => x.kind === "JUMP").length} parked then jumped`);
for (const x of top.sort((a, b) => a.t - b.t)) console.log(x.kind === "POP" ? `  POP  ${x.t.toFixed(2)}s  ${x.k}  ${x.gone ? "vanishes" : "collapses"} in one frame, back ${x.n} frame(s) later` : `  JUMP ${x.t.toFixed(2)}s  ${x.k}  moves ${x.n} px in one frame from rest`);
process.exit(top.length ? 1 : 0);
