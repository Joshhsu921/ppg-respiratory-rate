/* Compare estimate() option sets on BUT PPG development subjects AND the synthetic regression cases.
   A candidate is acceptable only if every synthetic case still passes. Usage: node scripts/web_dsp_grid.js */
const { execFileSync } = require('child_process');
const path = require('path');
const H = require('./butppg_dev.js');

function syntheticFailures(opts) {
  try {
    execFileSync(process.execPath, [path.join(__dirname, '..', 'tests', 'web_dsp_cases.js')],
      { env: { ...process.env, PPG_OPTS: JSON.stringify(opts) }, stdio: 'pipe' });
    return 0;
  } catch (e) {
    const m = String(e.stdout).match(/(\d+) FAILED/);
    return m ? +m[1] : -1;
  }
}

const grid = [];
for (const specLo of [0.5, 0.7]) for (const lo of [0.5, 0.7]) for (const minDistFactor of [0.45, 0.5, 0.6]) for (const subharmonicCheck of [false, true])
  grid.push({ specLo, lo, minDistFactor, subharmonicCheck });
console.log(`${H.recs.length} development recordings; synthetic failures in brackets`);
for (const o of grid) {
  const fails = syntheticFailures(o);
  H.evaluate(`[${fails} fail] specLo ${o.specLo} lo ${o.lo} minDist ${o.minDistFactor}${o.subharmonicCheck ? ' sub' : ''}`, o);
}
