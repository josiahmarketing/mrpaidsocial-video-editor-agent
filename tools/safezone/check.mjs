// check.mjs <index.html> [--bands "t,b,l,r;t,b,l,r"] [--box l,t,r,b] [--step 0.2] [--skip sel,sel] — prove that a
// HyperFrames composition keeps its text and cards inside the platform safe zone. Loads the page, seeks the GSAP
// timeline, and at every step measures each visible, settled element that carries text (or is a .card / .chip / img /
// svg inside the stage but outside the camera). Anything crossing the zone is listed with its time and rectangle.
// Elements still moving (a slide-in) are skipped until they rest.
//
// The zone is a stack of horizontal bands, each "top,bottom,left,right" in px on a 1080x1920 frame. An element must
// start below the first band's top, end above the last band's bottom, and stay between left and right of every band
// it touches. Default = the area inside ALL THREE published ad safe zones (TikTok's own file scaled from 720x1280:
// 240 top / 660 bottom / 120 sides plus the button column, nothing right of x 780 below y 840; Meta Reels: 14 % top /
// 35 % bottom / 6 % sides; YouTube Shorts: 288 top / 672 bottom / 48 left / 192 right):
//   y 288-840: x 120-888, then y 840-1248: x 120-780.
// --box l,t,r,b is the single-rectangle form (one platform, or a house box).
import { createRequire } from "node:module"; import { resolve } from "node:path";
const require = createRequire(import.meta.url); const { chromium } = require("playwright");
const args = process.argv.slice(2); const file = resolve(args[0]); const opt = (k, d) => { const i = args.indexOf(k); return i > 0 ? args[i + 1] : d; };
const step = +opt("--step", 0.2); const skip = opt("--skip", "#cam,#open,#wp,#bg2,#sheet,.door");
let bands = opt("--bands", "288,840,120,888;840,1248,120,780").split(";").map((b) => b.split(",").map(Number));
if (opt("--box", null)) { const [l, t, r, b] = opt("--box").split(",").map(Number); bands = [[t, b, l, r]]; }
if (bands.some((b) => b.length !== 4 || b.some(Number.isNaN))) { console.error("bad --bands / --box"); process.exit(2); }
const crossed = (e) => { const out = []; if (e.y0 < bands[0][0] - 0.5) out.push("top"); if (e.y1 > bands.at(-1)[1] + 0.5) out.push("bottom");
  for (const [t, b, l, r] of bands) { if (e.y1 <= t || e.y0 >= b) continue; if (e.x0 < l - 0.5 && !out.includes("left")) out.push("left"); if (e.x1 > r + 0.5 && !out.includes("right")) out.push("right"); }
  return out; };
const br = await chromium.launch(); const pg = await br.newPage({ viewport: { width: 1080, height: 1920 } }); await pg.goto("file://" + file); await pg.waitForFunction(() => window.__timelines && Object.keys(window.__timelines).length); await pg.evaluate(() => document.fonts.ready);
const dur = await pg.evaluate(() => Object.values(window.__timelines)[0].duration());
const measure = (tt) => pg.evaluate(([tt, skip]) => { const tl = Object.values(window.__timelines)[0]; tl.seek(tt, false); const out = [];
  const vis = (e) => { let o = 1; for (let n = e; n && n.nodeType === 1; n = n.parentElement) { const s = getComputedStyle(n); if (s.visibility === "hidden" || s.display === "none") return 0; o *= +s.opacity; } return o; };
  for (const e of document.querySelectorAll("#stage *")) { if (e.closest(skip) || ["SCRIPT", "STYLE", "AUDIO", "VIDEO"].includes(e.tagName)) continue;
    const own = [...e.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim()); const boxy = e.matches(".card,.chip,img,svg");
    if (!own && !boxy) continue; if (vis(e) < 0.9) continue; const q = e.getBoundingClientRect(); if (q.width < 2 || q.height < 2) continue;
    if (q.right <= 0 || q.left >= innerWidth || q.bottom <= 0 || q.top >= innerHeight) continue;   // fully off-screen: not shown
    out.push({ id: e.id || (e.closest("[id]")?.id + ">" + e.tagName.toLowerCase()), x0: q.left, y0: q.top, x1: q.right, y1: q.bottom, tx: (e.textContent || "").trim().slice(0, 28) }); }
  return out; }, [tt, skip]);
const bad = new Map(); let prev = new Map(); let n = 0;
for (let tt = 0; tt <= dur; tt = +(tt + step).toFixed(3)) { const cur = new Map((await measure(tt)).map((e) => [e.id + "|" + e.tx, e]));
  for (const [k, e] of cur) { const p = prev.get(k); if (!p || Math.abs(p.x0 - e.x0) + Math.abs(p.y0 - e.y0) + Math.abs(p.x1 - e.x1) + Math.abs(p.y1 - e.y1) > 1.5) continue; n++;
    const c = crossed(e); if (c.length) { const v = bad.get(k) || { ...e, from: tt, to: tt, edges: c }; v.to = tt; bad.set(k, v); } }
  prev = cur; }
await br.close();
console.log(`safe zone ${bands.map(([t, b, l, r]) => `y ${t}..${b}: x ${l}..${r}`).join("; ")} — ${n} settled element samples over ${dur.toFixed(1)} s, ${bad.size} outside`);
for (const v of bad.values()) console.log(`  OUT ${v.from.toFixed(1)}-${v.to.toFixed(1)}s  ${v.id}  "${v.tx}"  x ${v.x0.toFixed(0)}..${v.x1.toFixed(0)}  y ${v.y0.toFixed(0)}..${v.y1.toFixed(0)}  (${v.edges.join(", ")})`);
process.exit(bad.size ? 1 : 0);
