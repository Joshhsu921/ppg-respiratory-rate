/* Development harness for smartphone heart rate: BUT PPG development subjects only
   (data/processed/butppg_split.csv), fingertip at rest. Compares estimate() option sets.
   Never reads test subjects. Usage: node scripts/butppg_dev.js */
const fs = require('fs');
const path = require('path');
const D = require('../web/ppg-dsp.js');

const raw = fs.readdirSync('data/raw/butppg').map(d => path.join('data/raw/butppg', d)).find(d => fs.statSync(d).isDirectory());
const csv = f => fs.readFileSync(f, 'utf8').replace(/^﻿/, '').trim().split(/\r?\n/).map(l => l.split(','));
const ann = Object.fromEntries(csv(path.join(raw, 'quality-hr-ann.csv')).slice(1).map(r => [r[0], { q: +r[1], hr: +r[2] }]));
const info = Object.fromEntries(csv(path.join(raw, 'subject-info.csv')).slice(1).map(r => [r[0], { site: +r[5], motion: r[6] }]));
const split = Object.fromEntries(csv('data/processed/butppg_split.csv').slice(1).map(r => [r[0], r[1]]));

const recs = [];
for (const f of csv('data/processed/butppg_ppg.csv')) {
  const id = f[0];
  if (split[id.slice(0, 3)] !== 'dev' || info[id].site !== 1 || info[id].motion !== '0') continue;
  const rate = +f[1], n = +f[2];
  recs.push({ id, rate, red: f.slice(3, 3 + n).map(Number), green: f.slice(3 + n, 3 + 2 * n).map(v => (v === '' ? NaN : +v)), ...ann[id] });
}

function runOne(r, opts) {
  const t = r.red.map((_, i) => i / r.rate);
  const go = sig => { const o = D.estimate(t, sig, { fs: r.rate, win: r.red.length / r.rate, ...opts }); return o; };
  let best = go(r.red);
  if (r.green.every(Number.isFinite)) best = D.better(best, go(r.green));
  return best;
}

function evaluate(name, opts) {
  const err = [], ratio = [];
  let good = 0, goodErr = [];
  for (const r of recs) {
    const o = runOne(r, opts);
    const hr = o && Number.isFinite(o.hr) ? o.hr : NaN;
    if (!Number.isFinite(hr)) continue;
    err.push(Math.abs(hr - r.hr)); ratio.push(hr / r.hr);
    if (o.quality === 'good') { good++; goodErr.push(Math.abs(hr - r.hr)); }
  }
  const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
  const within = err.filter(e => e <= 5).length / recs.length;
  const half = ratio.filter(x => x > 0.4 && x < 0.6).length;
  console.log(`${name.padEnd(46)} MAE ${mean(err).toFixed(2).padStart(6)}  within5 ${(within * 100).toFixed(0).padStart(3)}%  halved ${String(half).padStart(3)}  `
    + `web-good ${String(good).padStart(3)} (MAE ${goodErr.length ? mean(goodErr).toFixed(2) : '--'})`);
}

module.exports = { recs, runOne, evaluate };
if (require.main === module) {
  console.log(`${recs.length} development recordings (fingertip, at rest)`);
  evaluate('current algorithm', {});
  const diag = { sub: 0, ok: 0 };
  for (const r of recs) {
    const o = runOne(r, {});
    if (!o) continue;
    const q = o.hrSpec / r.hr;
    if (q > 0.4 && q < 0.6) diag.sub++; else if (q > 0.9 && q < 1.1) diag.ok++;
  }
  console.log(`spectral HR about half the true rate: ${diag.sub}, about right: ${diag.ok}`);
}
