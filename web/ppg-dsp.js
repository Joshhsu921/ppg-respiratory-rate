/* Browser port of src/ppgrr/dsp.py: respiratory rate from a PPG signal.
   Works in the browser (window.PPGDSP) and in Node (require) for tests. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.PPGDSP = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  const RR_MIN = 4, RR_MAX = 42, RESP_FS = 4;

  // Irregular (t, v) samples -> uniform grid on [t0, t1)
  function resample(ts, vs, fs, t0, t1) {
    const n = Math.max(0, Math.floor((t1 - t0) * fs));
    const out = new Float64Array(n);
    let j = 0;
    for (let i = 0; i < n; i++) {
      const t = t0 + i / fs;
      while (j < ts.length - 2 && ts[j + 1] < t) j++;
      const a = ts[j], b = ts[j + 1] ?? a;
      const w = b > a ? Math.min(1, Math.max(0, (t - a) / (b - a))) : 0;
      out[i] = vs[j] + ((vs[j + 1] ?? vs[j]) - vs[j]) * w;
    }
    return out;
  }

  // RBJ Butterworth biquad coefficients [b0, b1, b2, a1, a2]
  function biquad(type, fc, fs) {
    const w = 2 * Math.PI * fc / fs, c = Math.cos(w), alpha = Math.sin(w) / (2 * Math.SQRT1_2);
    const a0 = 1 + alpha;
    const b = type === 'lp' ? [(1 - c) / 2, 1 - c, (1 - c) / 2] : [(1 + c) / 2, -(1 + c), (1 + c) / 2];
    return [b[0] / a0, b[1] / a0, b[2] / a0, (-2 * c) / a0, (1 - alpha) / a0];
  }
  function runBiquad(x, k) {
    const y = new Float64Array(x.length);
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let i = 0; i < x.length; i++) {
      const v = k[0] * x[i] + k[1] * x1 + k[2] * x2 - k[3] * y1 - k[4] * y2;
      x2 = x1; x1 = x[i]; y2 = y1; y1 = v; y[i] = v;
    }
    return y;
  }
  // Zero-phase filtering with reflection padding (like scipy filtfilt)
  function filtfilt(x, sections, padLen) {
    const n = x.length, p = Math.min(padLen, n - 1);
    if (n < 4) return Float64Array.from(x);
    const ext = new Float64Array(n + 2 * p);
    for (let i = 0; i < p; i++) { ext[i] = 2 * x[0] - x[p - i]; ext[n + p + i] = 2 * x[n - 1] - x[n - 2 - i]; }
    ext.set(x, p);
    const mean = ext.reduce((s, v) => s + v, 0) / ext.length;
    let y = ext.map(v => v - mean);
    for (const k of sections) y = runBiquad(y, k);
    y.reverse();
    for (const k of sections) y = runBiquad(y, k);
    y.reverse();
    return y.slice(p, p + n).map(v => v + (sections.some(s => s.hp) ? 0 : mean));
  }
  function bandpass(x, lo, hi, fs) {
    const hp = biquad('hp', lo, fs); hp.hp = true;
    return filtfilt(x, [hp, hp, biquad('lp', hi, fs), biquad('lp', hi, fs)], Math.round(fs * 3));
  }
  function lowpass(x, hi, fs) {
    return filtfilt(x, [biquad('lp', hi, fs), biquad('lp', hi, fs)], Math.round(fs * 3));
  }
  function detrend(x) {
    const n = x.length, mx = (n - 1) / 2;
    let my = 0; for (const v of x) my += v; my /= n;
    let sxy = 0, sxx = 0;
    for (let i = 0; i < n; i++) { sxy += (i - mx) * (x[i] - my); sxx += (i - mx) ** 2; }
    const slope = sxx ? sxy / sxx : 0;
    return x.map((v, i) => v - my - slope * (i - mx));
  }
  const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
  const std = a => { const m = mean(a); return Math.sqrt(mean(a.map(v => (v - m) ** 2))); };
  const median = a => { const s = [...a].sort((p, q) => p - q), h = s.length >> 1; return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2; };

  function findPeaks(x, minDist, threshold) {
    const cand = [];
    for (let i = 1; i < x.length - 1; i++) if (x[i] > threshold && x[i] >= x[i - 1] && x[i] > x[i + 1]) cand.push(i);
    cand.sort((a, b) => x[b] - x[a]);           // keep the tallest peaks, enforce spacing
    const taken = [];
    for (const i of cand) if (taken.every(j => Math.abs(i - j) >= minDist)) taken.push(i);
    return taken.sort((a, b) => a - b);
  }

  // Dominant breathing frequency of a 4 Hz series -> [breaths/min, share of band power]
  function spectralRR(y) {
    let s = detrend(y);
    s = bandpass(s, RR_MIN / 60, RR_MAX / 60, RESP_FS);
    const n = s.length;
    s = s.map((v, i) => v * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1))));
    let best = -1, bestRR = NaN, total = 0; const pw = [];
    for (let rr = RR_MIN; rr <= RR_MAX + 1e-9; rr += 0.1) {
      const f = rr / 60; let re = 0, im = 0;
      for (let i = 0; i < n; i++) { const ph = 2 * Math.PI * f * i / RESP_FS; re += s[i] * Math.cos(ph); im -= s[i] * Math.sin(ph); }
      const p = re * re + im * im; pw.push([rr, p]); total += p;
      if (p > best) { best = p; bestRR = rr; }
    }
    const near = pw.filter(([rr]) => Math.abs(rr - bestRR) <= 3).reduce((a, [, p]) => a + p, 0);
    return [bestRR, total > 0 ? near / total : NaN];
  }

  function seriesToResp(times, values, t0, t1) {
    if (times.length < 8) return null;
    const a = Math.max(t0, times[0]), b = Math.min(t1, times[times.length - 1]);
    if (b - a < 10) return null;
    return resample(times, values, RESP_FS, a, b);
  }

  /* ts: seconds, vs: raw PPG values (higher = more blood), fs: rate to resample to.
     Uses the last `win` seconds. */
  function estimate(ts, vs, { fs = 30, win = 32 } = {}) {
    const t1 = ts[ts.length - 1], t0 = Math.max(ts[0], t1 - win);
    const x = resample(ts, vs, fs, t0, t1);
    if (x.length < fs * 8) return null;
    const band = bandpass(x, 0.5, 4, fs);
    const low = lowpass(x, 4, fs);
    const pk = findPeaks(band, Math.round(fs * 60 / 180), 0.3 * std(band));
    const peaks = pk.slice(1), troughs = [];
    for (let i = 1; i < pk.length; i++) {
      let m = pk[i - 1];
      for (let j = pk[i - 1]; j < pk[i]; j++) if (band[j] < band[m]) m = j;
      troughs.push(m);
    }
    const bt = peaks.map(i => t0 + i / fs);
    const ibi = bt.slice(1).map((t, i) => t - bt[i]);
    const out = { t0, t1, band, fs, beats: bt, hr: NaN, rr: NaN, rrs: {}, resp: null, quality: 'poor' };
    if (ibi.length < 5) return out;
    out.hr = 60 / median(ibi);
    const cv = std(ibi) / mean(ibi);

    const medIbi = median(ibi);
    const fmKeep = ibi.map((v, i) => [(bt[i] + bt[i + 1]) / 2, v]).filter(([, v]) => Math.abs(v - medIbi) <= 0.3 * medIbi);
    const series = {
      riiv: [bt, peaks.map(i => low[i])],
      riav: [bt, peaks.map((i, k) => low[i] - low[troughs[k]])],
      rifv: [fmKeep.map(p => p[0]), fmKeep.map(p => p[1])],
    };
    for (const [name, [tt, vv]] of Object.entries(series)) {
      const y = seriesToResp(tt, vv, t0, t1);
      out.rrs[name] = y ? spectralRR(y)[0] : NaN;
      if (name === 'riiv' && y) out.resp = bandpass(detrend(y), RR_MIN / 60, RR_MAX / 60, RESP_FS);
    }
    const est = Object.values(out.rrs);
    if (est.every(Number.isFinite) && std(est) <= 4) out.rr = mean(est);  // smart fusion
    out.quality = cv < 0.2 && Number.isFinite(out.rr) ? 'good' : cv < 0.35 ? 'fair' : 'poor';
    return out;
  }

  // Same test signal as ppgrr.dsp.synthetic_ppg
  function synthetic(duration = 64, fs = 30, hr = 72, rr = 15, noise = 0.02, seed = 1) {
    let s = seed; const rand = () => { s = (s * 16807) % 2147483647; return s / 2147483647 - 0.5; };
    const ts = [], vs = []; let phase = 0;
    for (let i = 0; i < duration * fs; i++) {
      const t = i / fs, resp = Math.sin(2 * Math.PI * rr / 60 * t);
      phase += 2 * Math.PI * hr / 60 * (1 + 0.05 * resp) / fs;
      const pulse = Math.sin(phase) + 0.4 * Math.sin(2 * phase - 1);
      ts.push(t); vs.push((1 + 0.15 * resp) * pulse + 0.3 * resp + noise * 2 * rand());
    }
    return { ts, vs };
  }

  return { estimate, synthetic, spectralRR, bandpass, lowpass, resample, RR_MIN, RR_MAX };
});
