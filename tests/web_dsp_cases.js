/* Heart-rate cases for web/ppg-dsp.js on synthetic camera-like PPG (30 Hz): dicrotic waves of
   different size and position, noise, fast and slow rates, and a weak every-second beat.
   Run: node tests/web_dsp_cases.js  (exits non-zero on failure) */
const path = require('path');
const D = require(path.join(__dirname, '..', 'web', 'ppg-dsp.js'));
function gen({ rr = 15, hr = 75, noise = 0.05, diaPos = 0.30, diaAmp = 0.45, weakEvery = 0, weakAmp = 0.15, seed = 7 }) {
  let s = seed; const rnd = () => { s = (s * 16807) % 2147483647; return s / 2147483647 - 0.5; };
  const ts = [], vs = []; let ph = 0;
  for (let i = 0; i < 40 * 30; i++) {
    const t = i / 30, resp = Math.sin(2 * Math.PI * rr / 60 * t); ph += 2 * Math.PI * hr / 60 * (1 + 0.05 * resp) / 30;
    const n = Math.floor(ph / (2 * Math.PI)), c = (ph / (2 * Math.PI)) % 1; const amp = weakEvery && n % weakEvery === 1 ? weakAmp : 1;
    const pulse = amp * (Math.exp(-Math.pow((c - 0.15) / 0.07, 2)) + diaAmp * Math.exp(-Math.pow((c - 0.15 - diaPos) / 0.08, 2)));
    ts.push(t); vs.push((1 + 0.15 * resp) * pulse + 0.25 * resp + noise * 2 * rnd());
  }
  return { ts, vs };
}
const cases = [
  ['normal, dicrotic at 30%', {}, 75], ['normal, dicrotic at 40%', { diaPos: 0.40 }, 75], ['strong dicrotic at 45%', { diaPos: 0.45, diaAmp: 0.6 }, 75],
  ['strong dicrotic at 40%, HR 60', { hr: 60, diaPos: 0.40, diaAmp: 0.7 }, 60], ['strong dicrotic 35%, HR 90', { hr: 90, diaPos: 0.35, diaAmp: 0.8 }, 90],
  ['noisy normal', { noise: 0.2 }, 75], ['fast HR 120', { hr: 120 }, 120], ['fast HR 150', { hr: 150 }, 150], ['slow HR 45', { hr: 45 }, 45],
  ['weak every 2nd beat (true 80)', { hr: 80, weakEvery: 2, weakAmp: 0.3, diaAmp: 0.2 }, 80],
];
let fail = 0;
for (const seed of [7, 11, 23]) for (const [name, o, truth] of cases) {
  const g = gen({ ...o, seed }); const r = D.estimate(g.ts, g.vs, { fs: 30, win: 32 });
  const ok = Math.abs(r.hr - truth) < 3 || (r.alternans && Math.abs(r.hrAlt - truth) < 3); if (!ok) fail++;
  if (seed === 7 || !ok) console.log((ok ? 'ok  ' : 'FAIL') + ` seed ${seed} ` + name.padEnd(32), 'truth', truth, 'hr', r.hr.toFixed(1), 'doubling', r.doubling, 'alt', r.alternans ? r.hrAlt.toFixed(0) : '-', 'q', r.quality, 'rr', Number.isFinite(r.rr) ? r.rr.toFixed(1) : 'NaN');
}
const s = D.synthetic(40, 30, 72, 15); const r = D.estimate(s.ts, s.vs); console.log('original synthetic', r.hr.toFixed(1), r.rr.toFixed(1), r.quality);
console.log(fail ? `${fail} FAILED` : 'all passed'); process.exit(fail ? 1 : 0);
