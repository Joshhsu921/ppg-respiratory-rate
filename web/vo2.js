/* Experimental VO2 max card. Shows two estimates with their uncertainty against age- and sex-specific
   norms, using the latest saved measurement taken at rest, sitting or lying.
   - Demographic model: linear regression on age, sex and BMI, trained on 955 treadmill tests of athletes
     (data/vo2max_model.json, scripts/vo2max.py); range = cross-validated 2.5–97.5 % error.
   - Uth et al. 2004 heart-rate ratio: 15.3 x (208 - 0.7 x age) / resting HR. Not validated by this project.
   Norms: FRIEND registry treadmill percentiles, Kaminsky et al., Mayo Clin Proc 2015;90:1515-23, Table 2. */
(function () {
  const PROFILE_KEY = 'ppgrr.profile.v1';
  // [age from, age to, P5, P10, P25, P50, P75, P90, P95] in mL/kg/min
  const FRIEND = {
    m: [[20, 29, 29.0, 32.1, 40.1, 48.0, 55.2, 61.8, 66.3], [30, 39, 27.2, 30.2, 35.9, 42.4, 49.2, 56.5, 59.8],
        [40, 49, 24.2, 26.8, 31.9, 37.8, 45.0, 52.1, 55.6], [50, 59, 20.9, 22.8, 27.1, 32.6, 39.7, 45.6, 50.7],
        [60, 69, 17.4, 19.8, 23.7, 28.2, 34.5, 40.3, 43.0], [70, 79, 16.3, 17.1, 20.4, 24.4, 30.4, 36.6, 39.7]],
    f: [[20, 29, 21.7, 23.9, 30.5, 37.6, 44.7, 51.3, 56.0], [30, 39, 19.0, 20.9, 25.3, 30.2, 36.1, 41.4, 45.8],
        [40, 49, 17.0, 18.8, 22.1, 26.7, 32.4, 38.4, 41.7], [50, 59, 16.0, 17.3, 19.9, 23.4, 27.6, 32.0, 35.9],
        [60, 69, 13.4, 14.6, 17.2, 20.0, 23.8, 27.0, 29.4], [70, 79, 13.1, 13.6, 15.6, 18.3, 20.8, 23.1, 24.1]],
  };
  const PCTS = [5, 10, 25, 50, 75, 90, 95];
  // levels by percentile, for display only
  // < P25, P25–50, P50–75, P75–90, >= P90; short names so they fit under the scale on a phone
  const LEVELS = [[0, 25, '偏低'], [25, 50, '中下'], [50, 75, '中上'], [75, 90, '良好'], [90, 100, '優秀']];
  const RESTING = e => e.context === '休息' && (e.posture === '坐著' || e.posture === '躺著') && Number.isFinite(e.hr);
  const $ = id => document.getElementById(id);
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  let model = null, lastEntries = [];

  function loadProfile() {
    try { const p = JSON.parse(localStorage.getItem(PROFILE_KEY) || 'null'); return p && typeof p === 'object' ? p : null; }
    catch (_) { return null; }
  }
  function saveProfile(p) { try { localStorage.setItem(PROFILE_KEY, JSON.stringify(p)); return true; } catch (_) { return false; } }

  function normRow(sex, age) {
    const row = (FRIEND[sex] || []).find(r => age >= r[0] && age <= r[1] + 0.999);
    return row ? row.slice(2) : null;
  }
  // percentile of a value within a norm row, linear between the published points
  function percentile(v, row) {
    if (v <= row[0]) return { p: 5, below: true };
    if (v >= row[6]) return { p: 95, above: true };
    for (let i = 0; i < 6; i++) if (v <= row[i + 1]) return { p: PCTS[i] + (PCTS[i + 1] - PCTS[i]) * (v - row[i]) / (row[i + 1] - row[i]) };
    return { p: 95 };
  }
  const level = p => LEVELS.find(([a, b]) => p >= a && p < b)?.[2] ?? LEVELS[LEVELS.length - 1][2];
  // value at a given percentile (for drawing level boundaries)
  function valueAt(p, row) {
    if (p <= 5) return row[0]; if (p >= 95) return row[6];
    for (let i = 0; i < 6; i++) if (p <= PCTS[i + 1]) return row[i] + (row[i + 1] - row[i]) * (p - PCTS[i]) / (PCTS[i + 1] - PCTS[i]);
    return row[6];
  }

  function estimates(profile, hrRest) {
    const bmi = profile.weight / Math.pow(profile.height / 100, 2);
    const female = profile.sex === 'f' ? 1 : 0;
    const out = { bmi };
    if (model && (profile.sex === 'm' || profile.sex === 'f')) {   // the model needs sex
      const v = model.intercept + model.coef.age * profile.age + model.coef.female * female + model.coef.bmi * bmi;
      out.demo = { v, lo: v + model.residual_p2_5, hi: v + model.residual_p97_5 };
    }
    if (Number.isFinite(hrRest)) out.uth = 15.3 * (208 - 0.7 * profile.age) / hrRest;
    return out;
  }

  /* ---------- rendering ---------- */
  const el = (tag, attrs, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === 'class') n.className = v; else if (k.startsWith('on')) n.addEventListener(k.slice(2), v); else n.setAttribute(k, v === true ? '' : v);
    }
    for (const c of kids.flat()) if (c != null && c !== false) n.append(c.nodeType ? c : String(c));
    return n;
  };
  const f1 = v => Number.isFinite(v) ? v.toFixed(1) : '--';
  const when = iso => { const d = new Date(iso); return `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`; };

  function profileForm(p) {
    const box = $('vo2Profile');
    box.hidden = false;
    $('pAge').value = p?.age ?? ''; $('pHeight').value = p?.height ?? ''; $('pWeight').value = p?.weight ?? '';
    for (const b of box.querySelectorAll('[data-sex]')) b.setAttribute('aria-pressed', String(p?.sex === b.dataset.sex));
  }

  function drawScale(row, est) {
    const cv = $('cvVo2');
    const r = cv.getBoundingClientRect(), d = devicePixelRatio || 1;
    cv.width = Math.round(r.width * d); cv.height = Math.round(r.height * d);
    const g = cv.getContext('2d'); g.setTransform(d, 0, 0, d, 0, 0);
    const w = r.width, h = r.height;
    g.clearRect(0, 0, w, h);
    const vals = [row[0] - 4, row[6] + 4, est.demo?.lo, est.demo?.hi, est.uth].filter(Number.isFinite);
    const lo = Math.floor(Math.min(...vals) / 5) * 5, hi = Math.ceil(Math.max(...vals) / 5) * 5;
    const L = 8, R = 8, X = v => L + (v - lo) / (hi - lo) * (w - L - R);
    // level bands
    const bandY = 46, bandH = 16;
    LEVELS.forEach(([a, b, name], i) => {
      const x0 = a === 0 ? X(lo) : X(valueAt(a, row)), x1 = b === 100 ? X(hi) : X(valueAt(b, row));
      g.fillStyle = css('--lvl-' + i); g.fillRect(x0, bandY, x1 - x0 - 1, bandH);   // 1 px gap between levels
      g.fillStyle = css('--ink'); g.font = '11px ' + css('--f-body'); g.textAlign = 'center';
      if (x1 - x0 > 28) g.fillText(name, (x0 + x1) / 2, bandY + bandH + 14);
    });
    // axis ticks
    g.fillStyle = css('--muted'); g.font = '11px ' + css('--f-mono'); g.textAlign = 'center';
    for (let v = lo; v <= hi; v += 10) g.fillText(v, X(v), h - 4);
    // demographic range
    if (est.demo) {
      g.strokeStyle = css('--resp'); g.lineWidth = 6; g.lineCap = 'round';
      g.beginPath(); g.moveTo(X(est.demo.lo), 22); g.lineTo(X(est.demo.hi), 22); g.stroke();
      g.fillStyle = css('--panel'); g.strokeStyle = css('--resp'); g.lineWidth = 2;
      g.beginPath(); g.arc(X(est.demo.v), 22, 6, 0, 2 * Math.PI); g.fill(); g.stroke();
    }
    if (Number.isFinite(est.uth)) {
      const x = X(est.uth);
      g.fillStyle = css('--pleth');
      g.beginPath(); g.moveTo(x, 30); g.lineTo(x - 6, 40); g.lineTo(x + 6, 40); g.closePath(); g.fill();
    }
  }

  function trend(entries) {
    const rest = entries.filter(RESTING).slice(-12);
    if (rest.length < 2) return rest.length ? el('p', { class: 'status' }, '再量一次休息時的心跳,就能看到變化趨勢。') : null;
    const first = rest[0], last = rest[rest.length - 1];
    const diff = last.hr - first.hr;
    const lo = Math.min(...rest.map(e => e.hr)) - 3, hi = Math.max(...rest.map(e => e.hr)) + 3;
    const pts = rest.map((e, i) => `${(i / (rest.length - 1) * 100).toFixed(1)},${(30 - (e.hr - lo) / (hi - lo) * 28).toFixed(1)}`).join(' ');
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 100 32'); svg.setAttribute('preserveAspectRatio', 'none'); svg.setAttribute('class', 'spark');
    svg.setAttribute('aria-hidden', 'true');
    const pl = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    pl.setAttribute('points', pts); pl.setAttribute('fill', 'none'); pl.setAttribute('stroke', 'currentColor');
    pl.setAttribute('stroke-width', '1.5'); pl.setAttribute('vector-effect', 'non-scaling-stroke');
    svg.append(pl);
    return el('div', { class: 'trendrow' },
      el('div', null, el('div', { class: 'k' }, `靜止心跳趨勢(最近 ${rest.length} 筆)`),
        el('div', { class: 'sub' }, `${when(first.t)} ${first.hr} → ${when(last.t)} ${last.hr} 次/分(${diff > 0 ? '+' : ''}${diff})`)),
      svg);
  }

  function render(entries) {
    lastEntries = entries || lastEntries;
    const body = $('vo2Body');
    if (!body) return;
    const profile = loadProfile();
    const latest = [...lastEntries].reverse().find(RESTING);
    $('vo2Profile').hidden = !!profile && $('vo2Profile').dataset.editing !== '1';
    if (!profile) { profileForm(null); }
    if (!profile) {
      body.replaceChildren(el('p', { class: 'status' }, '先填寫上方的基本資料。資料只存在這台裝置上。'));
      return;
    }
    const age = +profile.age;
    if (age < 18) {   // intended for adults only: see the page header and README
      body.replaceChildren(el('p', { class: 'notice warn' }, '本工具適用 18 歲以上成人。兒童的呼吸、心跳範圍和體能常模都不同,所以不計算 VO2 max。'));
      return;
    }
    const row = profile.sex === 'f' || profile.sex === 'm' ? normRow(profile.sex, age) : null;
    const est = estimates(profile, latest?.hr);
    const warnings = [];
    if (model && age > model.train.age_max) warnings.push(`估算模型的訓練資料年齡為 ${Math.round(model.train.age_min)}–${model.train.age_max} 歲,你的年齡在範圍外。`);
    if (age >= 40) warnings.push('模型以運動員訓練,年齡幾乎沒有影響;一般人每 10 年約下降 10%,所以估算值可能明顯偏高。');
    if (profile.sex === 'x') warnings.push('基本資料模型和常模都需要性別,所以只顯示心跳比值公式,不顯示等級。');
    if (!row && profile.sex !== 'x') warnings.push('常模只涵蓋 20–79 歲,無法顯示等級。');

    const lines = [];
    if (est.demo) {
      const pr = row && [percentile(est.demo.lo, row), percentile(est.demo.hi, row)];
      const levelsCrossed = row ? LEVELS.filter(([a, b]) => b > pr[0].p && a < pr[1].p).length : 0;
      lines.push(el('div', { class: 'est' },
        el('i', { class: 'mk mk-demo', 'aria-hidden': 'true' }),
        el('div', null,
          el('div', null, el('b', null, '基本資料模型:'), ` 約 ${f1(est.demo.v)},95% 範圍 ${f1(est.demo.lo)}–${f1(est.demo.hi)} mL/kg/min`),
          el('div', { class: 'sub' }, `年齡、性別、BMI ${f1(est.bmi)}。依 955 次跑步機實測交叉驗證,平均誤差 ${model.cv_mae}。` +
            (row ? `範圍橫跨 ${levelsCrossed} 個等級。` : '')))));
    }
    if (Number.isFinite(est.uth)) {
      const pu = row ? percentile(est.uth, row) : null;
      lines.push(el('div', { class: 'est' },
        el('i', { class: 'mk mk-uth', 'aria-hidden': 'true' }),
        el('div', null,
          el('div', null, el('b', null, '心跳比值公式:'), ` 約 ${f1(est.uth)} mL/kg/min` + (pu ? `,約第 ${Math.round(pu.p)} 百分位(${level(pu.p)})` : '')),
          el('div', { class: 'sub' }, `用 ${when(latest.t)} 休息${latest.posture}時的心跳 ${latest.hr} 次/分${latest.example ? '(範例資料)' : latest.source === 'sim' ? '(模擬訊號)' : ''},最大心跳以 208 − 0.7 × 年齡估算。文獻公式(Uth 2004),本專案沒有資料可以驗證它的誤差,所以不顯示範圍。`))));
    } else {
      lines.push(el('p', { class: 'status' }, '需要一筆情境為「休息」、姿勢為「坐著」或「躺著」的量測,才能用你的靜止心跳估算。'));
    }

    body.replaceChildren(...[
      row ? el('div', { class: 'strip' }, el('span', null, `同年齡、同性別常模(FRIEND,${profile.sex === 'f' ? '女性' : '男性'} ${Math.floor(age / 10) * 10}–${Math.floor(age / 10) * 10 + 9} 歲)`),
        el('canvas', { id: 'cvVo2', style: 'height:96px' })) : null,
      el('div', { class: 'ests' }, lines),
      warnings.length ? el('ul', { class: 'warnlist' }, warnings.map(w => el('li', null, w))) : null,
      trend(lastEntries),
      el('p', { class: 'note' }, '實驗性功能:誤差範圍很大,不能取代運動心肺測試,也不能作為運動處方依據。長期追蹤自己的靜止心跳變化,通常比單次 VO2 max 數字更有參考價值。'),
    ].filter(Boolean));   // replaceChildren would print null as text
    if (row) requestAnimationFrame(() => drawScale(row, est));
  }

  function init() {
    fetch('data/vo2max_model.json').then(r => r.ok ? r.json() : null).then(m => { model = m; render(); }).catch(() => render());
    const box = $('vo2Profile');
    let sex = loadProfile()?.sex ?? null;
    for (const b of box.querySelectorAll('[data-sex]')) b.addEventListener('click', () => {
      sex = b.dataset.sex;
      for (const o of box.querySelectorAll('[data-sex]')) o.setAttribute('aria-pressed', String(o === b));
    });
    $('btnEditProfile').addEventListener('click', () => { box.dataset.editing = '1'; profileForm(loadProfile()); sex = loadProfile()?.sex ?? null; });
    box.addEventListener('submit', e => {
      e.preventDefault();
      const age = parseFloat($('pAge').value), height = parseFloat($('pHeight').value), weight = parseFloat($('pWeight').value);
      const bad = !(age >= 10 && age <= 100) ? '年齡' : !(height >= 120 && height <= 220) ? '身高' : !(weight >= 30 && weight <= 200) ? '體重' : !sex ? '性別' : null;
      if (bad) { $('profileMsg').textContent = `請確認${bad}。`; return; }
      $('profileMsg').textContent = saveProfile({ age, height, weight, sex }) ? '' : '無法儲存:這個瀏覽器封鎖了網站資料。';
      box.dataset.editing = '';
      render();
    });
    addEventListener('resize', () => render());
    render();
  }
  window.PPGVO2 = { init, render, isResting: RESTING };
})();
