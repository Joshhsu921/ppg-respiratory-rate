/* Run the deployed browser algorithm (web/ppg-dsp.js) on BIDMC PPG, resampled to 30 Hz like a phone camera.
   Usage: node scripts/hr_web_eval.js [csvDir] [out.csv]
   Writes per-window web estimates (heart rate, quality, respiratory rate, strong-weak alternation flag)
   for 32 s and 10 s windows. */
const fs = require('fs');
const path = require('path');
const D = require('../web/ppg-dsp.js');

function findCsvDir(root) {
  for (const e of fs.readdirSync(root, { withFileTypes: true })) {
    const p = path.join(root, e.name);
    if (e.isDirectory()) { const hit = findCsvDir(p); if (hit) return hit; }
    else if (e.name === 'bidmc_01_Signals.csv') return root;
  }
  return null;
}

const csvDir = process.argv[2] || findCsvDir('data/raw');
const out = process.argv[3] || 'results/hr_web_predictions.csv';
const WINDOWS = [[32, 8], [10, 4]];   // [length, step] in seconds
const FS = 30;

const rows = ['subject,start_s,win_s,web_hr,web_quality,web_rr,web_alternans,web_alternation'];
const subjects = fs.readdirSync(csvDir).filter(f => /^bidmc_\d+_Signals\.csv$/.test(f)).map(f => f.split('_')[1]).sort();
for (const s of subjects) {
  const lines = fs.readFileSync(path.join(csvDir, `bidmc_${s}_Signals.csv`), 'utf8').trim().split(/\r?\n/);
  const head = lines[0].split(',').map(c => c.trim());
  const ti = head.findIndex(c => c.startsWith('Time')), pi = head.indexOf('PLETH');
  const t = [], v = [];
  for (let i = 1; i < lines.length; i++) {
    const c = lines[i].split(',');
    const val = parseFloat(c[pi]);
    if (Number.isFinite(val)) { t.push(parseFloat(c[ti])); v.push(val); }
  }
  // 125 Hz -> 30 Hz uniform, as the camera pipeline sees it
  const t0 = t[0], t1 = t[t.length - 1];
  const x = D.resample(t, v, FS, t0, t1);
  const tx = Array.from(x, (_, i) => t0 + i / FS);
  for (const [win, step] of WINDOWS) {
    for (let start = 0; start + win <= t1 - t0 + 1e-9; start += step) {
      const a = Math.round(start * FS), b = Math.round((start + win) * FS);
      const r = D.estimate(tx.slice(a, b), Array.from(x.slice(a, b)), { fs: FS, win });
      const hr = r && Number.isFinite(r.hr) ? r.hr.toFixed(2) : '';
      const rr = r && Number.isFinite(r.rr) ? r.rr.toFixed(2) : '';
      const alt = r && r.sqi && Number.isFinite(r.sqi.alternation) ? r.sqi.alternation.toFixed(2) : '';
      rows.push([s, start, win, hr, r ? r.quality : '', rr, r && r.alternans ? 1 : 0, alt].join(','));
    }
  }
  process.stdout.write(`subject ${s} done\n`);
}
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, rows.join('\n'));
console.log(`wrote ${rows.length - 1} rows -> ${out}`);
