/* Run the deployed browser algorithm (web/ppg-dsp.js) on BUT PPG smartphone recordings.
   Input: data/processed/butppg_ppg.csv from scripts/butppg_export.py
          (id, fs, n, n red samples, n green samples; green is empty for single-channel records).
   As in the web demo, red and green are both estimated and the better result is kept
   (PPGDSP.better). A red-only run with the polarity flipped is kept as a sensitivity check.
   Output: results/butppg_web_predictions.csv.  Usage: node scripts/butppg_eval.js */
const fs = require('fs');
const path = require('path');
const D = require('../web/ppg-dsp.js');

const input = process.argv[2] || 'data/processed/butppg_ppg.csv';
const out = process.argv[3] || 'results/butppg_web_predictions.csv';
const rows = ['id,web_hr,web_quality,web_reason,web_channel,web_hr_flipped'];
for (const line of fs.readFileSync(input, 'utf8').trim().split(/\r?\n/)) {
  const f = line.split(',');
  const id = f[0], rate = parseFloat(f[1]), n = parseInt(f[2], 10);
  const red = f.slice(3, 3 + n).map(Number);
  const green = f.slice(3 + n, 3 + 2 * n).map(v => (v === '' ? NaN : Number(v)));
  const t = red.map((_, i) => i / rate);
  const run = sig => D.estimate(t, sig, { fs: rate, win: n / rate });
  let best = run(red);
  if (best) best.channel = 'red';
  if (green.every(Number.isFinite)) {
    const g = run(green);
    if (g) g.channel = 'green';
    best = D.better(best, g);
  }
  const flipped = run(red.map(x => -x));
  const hr = r => (r && Number.isFinite(r.hr) ? r.hr.toFixed(2) : '');
  rows.push([id, hr(best), best ? best.quality : '', best && best.reason ? best.reason : '',
             best ? best.channel : '', hr(flipped)].join(','));
}
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, rows.join('\n'));
console.log(`wrote ${rows.length - 1} rows -> ${out}`);
