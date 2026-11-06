/* Personal measurement log: saves each measurement with its context to this browser only
   (localStorage), draws the trend, summarises by context and compares with self-counts. */
(function () {
  const KEY = 'ppgrr.log.v1';
  const CONTEXTS = ['休息', '起床後', '睡前', '運動後', '工作或讀書', '壓力或焦慮', '身體不適', '其他'];
  const POSTURES = ['坐著', '躺著', '站著'];
  const QLABEL = { good: '良好', fair: '尚可', poor: '不穩定' };
  const $ = id => document.getElementById(id);
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const ctxColor = c => css('--ctx-' + Math.max(0, CONTEXTS.indexOf(c)));

  function load() {
    try { const v = JSON.parse(localStorage.getItem(KEY) || '[]'); return Array.isArray(v) ? v : []; }
    catch (_) { return []; }
  }
  function store(list) {
    try { localStorage.setItem(KEY, JSON.stringify(list)); return true; } catch (_) { return false; }
  }
  let entries = load();

  const el = (tag, attrs, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === 'class') n.className = v;
      else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? '' : v);
    }
    for (const c of kids.flat()) if (c != null && c !== false) n.append(c.nodeType ? c : String(c));
    return n;
  };
  const fmt = (v, d = 1) => Number.isFinite(v) ? v.toFixed(d) : '--';
  const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
  const when = iso => {
    const d = new Date(iso);
    return `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
  };

  /* ---------- save form ---------- */
  let snap = null, ctx = null, posture = null, timer = null;

  function chips(container, options, current, pick) {
    container.replaceChildren(...options.map(o => el('button', {
      type: 'button', class: 'chip', 'aria-pressed': String(o === current),
      onclick: () => pick(o === current ? null : o),
    }, container.id === 'ctxChips' ? el('i', { class: 'dot', style: `background:${ctxColor(o)}` }) : null, o)));
  }
  function renderForm() {
    chips($('ctxChips'), CONTEXTS, ctx, v => { ctx = v; renderForm(); });
    chips($('postureChips'), POSTURES, posture, v => { posture = v; renderForm(); });
    $('btnDoSave').disabled = !ctx;
    $('saveNeed').hidden = !!ctx;
  }
  function openForm(s) {
    snap = s; ctx = null; posture = null;
    $('manualCount').value = ''; $('note').value = ''; $('timerText').textContent = '';
    $('snap').replaceChildren(
      el('span', null, '呼吸 ', el('b', null, fmt(s.rr)), ' 次/分'),
      el('span', null, '心跳 ', el('b', null, fmt(s.hr, 0)), ' 次/分'),
      el('span', null, '品質 ', el('b', null, QLABEL[s.quality] || '--')),
      s.source === 'sim' ? el('span', { class: 'badge' }, '模擬訊號') : null);
    renderForm();
    $('saveForm').hidden = false;
    $('saveForm').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
  function closeForm() {
    clearInterval(timer); timer = null; $('btnTimer').disabled = false;
    $('saveForm').hidden = true; snap = null;
  }
  function startTimer() {
    let left = 30;
    $('btnTimer').disabled = true;
    $('timerText').textContent = '開始數!還剩 30 秒';
    timer = setInterval(() => {
      left--;
      if (left > 0) { $('timerText').textContent = `還剩 ${left} 秒`; return; }
      clearInterval(timer); timer = null; $('btnTimer').disabled = false;
      $('timerText').textContent = '時間到,輸入你數到的次數。';
      $('manualCount').focus();
    }, 1000);
  }
  function submit(e) {
    e.preventDefault();
    if (!snap || !ctx) return;
    const count = parseFloat($('manualCount').value);
    entries.push({
      id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
      t: new Date().toISOString(), source: snap.source,
      rr: +snap.rr.toFixed(1), hr: Number.isFinite(snap.hr) ? Math.round(snap.hr) : null,
      quality: snap.quality, channel: snap.channel || null, parts: snap.parts || null,
      context: ctx, posture, manual: Number.isFinite(count) && count > 0 ? count * 2 : null,
      note: $('note').value.trim().slice(0, 100),
    });
    const ok = store(entries);
    closeForm();
    render();
    const resting = window.PPGVO2 && window.PPGVO2.isResting(entries[entries.length - 1]);
    $('saveHint').textContent = !ok ? '無法儲存:這個瀏覽器封鎖了網站資料(例如私密瀏覽模式)。'
      : resting ? '已儲存。這是一筆休息時的量測,下方的 VO2 max 估計已更新。' : '已儲存到「我的紀錄」。';
  }

  /* ---------- log section ---------- */
  function tiles() {
    const real = entries;
    const last = real[real.length - 1];
    const rest = real.filter(e => e.context === '休息');
    const paired = real.filter(e => Number.isFinite(e.manual));
    const mae = paired.length ? mean(paired.map(e => Math.abs(e.rr - e.manual))) : NaN;
    const tile = (k, v, u, sub) => el('div', { class: 'ro' }, el('div', { class: 'k' }, k),
      el('div', null, el('span', { class: 'v sm' }, v), u ? el('span', { class: 'u' }, u) : null),
      el('div', { class: 'sub' }, sub));
    $('tiles').replaceChildren(
      tile('最近一次', last ? fmt(last.rr) : '--', '次/分', last ? `${when(last.t)} · ${last.context}` : '尚無紀錄'),
      tile('休息時平均', rest.length ? fmt(mean(rest.map(e => e.rr))) : '--', '次/分', `${rest.length} 筆 · 常見範圍 12–20`),
      tile('與自己數的差距', fmt(mae), '次/分', paired.length ? `平均絕對差距 · ${paired.length} 筆` : '儲存時自己數一次就會計算'));
  }
  function contextTable() {
    const groups = CONTEXTS.map(c => [c, entries.filter(e => e.context === c)]).filter(([, g]) => g.length);
    if (!groups.length) { $('ctxTable').replaceChildren(); return; }
    $('ctxTable').replaceChildren(el('table', { class: 'ctxt' },
      el('thead', null, el('tr', null, ['情境', '筆數', '平均呼吸', '平均心跳'].map(h => el('th', null, h)))),
      el('tbody', null, groups.map(([c, g]) => {
        const hrs = g.map(e => e.hr).filter(Number.isFinite);
        return el('tr', null,
          el('td', null, el('i', { class: 'dot', style: `background:${ctxColor(c)}` }), c),
          el('td', null, g.length), el('td', null, fmt(mean(g.map(e => e.rr)))),
          el('td', null, hrs.length ? fmt(mean(hrs), 0) : '--'));
      }))));
  }
  function drawTrend() {
    const cv = $('cvTrend');
    const r = cv.getBoundingClientRect(), d = devicePixelRatio || 1;
    cv.width = Math.round(r.width * d); cv.height = Math.round(r.height * d);
    const g = cv.getContext('2d'); g.setTransform(d, 0, 0, d, 0, 0);
    const w = r.width, h = r.height;
    g.clearRect(0, 0, w, h);
    if (!entries.length) return;
    const ts = entries.map(e => Date.parse(e.t));
    let t0 = Math.min(...ts), t1 = Math.max(...ts);
    if (t1 - t0 < 3600e3) { t0 -= 1800e3; t1 += 1800e3; }
    const vals = entries.flatMap(e => [e.rr, e.manual]).filter(Number.isFinite);
    const lo = Math.min(8, Math.floor(Math.min(...vals) - 2)), hi = Math.max(24, Math.ceil(Math.max(...vals) + 2));
    const L = 34, R = 10, T = 8, B = 22;
    const X = t => L + (t - t0) / (t1 - t0) * (w - L - R), Y = v => T + (hi - v) / (hi - lo) * (h - T - B);
    g.fillStyle = css('--band'); g.fillRect(L, Y(20), w - L - R, Y(12) - Y(20));
    g.font = '11px ' + css('--f-mono'); g.fillStyle = css('--muted'); g.strokeStyle = css('--line'); g.lineWidth = 1;
    for (let v = Math.ceil(lo / 4) * 4; v <= hi; v += 4) {
      g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(w - R, Y(v)); g.stroke(); g.fillText(v, 6, Y(v) + 4);
    }
    const days = (t1 - t0) / 86400e3;
    const ticks = 4;
    for (let i = 0; i <= ticks; i++) {
      const t = t0 + (t1 - t0) * i / ticks, dt = new Date(t);
      const label = days > 2 ? `${dt.getMonth() + 1}/${dt.getDate()}` : `${String(dt.getHours()).padStart(2, '0')}:${String(dt.getMinutes()).padStart(2, '0')}`;
      const x = Math.min(w - R - 30, Math.max(L, X(t) - 14));
      g.fillText(label, x, h - 6);
    }
    const order = entries.map((e, i) => [ts[i], e]).sort((a, b) => a[0] - b[0]);
    g.strokeStyle = css('--line'); g.lineWidth = 1.2; g.beginPath();
    order.forEach(([t, e], i) => (i ? g.lineTo(X(t), Y(e.rr)) : g.moveTo(X(t), Y(e.rr))));
    g.stroke();
    for (const [t, e] of order) {
      if (Number.isFinite(e.manual)) {           // self-count: hollow ring
        g.strokeStyle = ctxColor(e.context); g.lineWidth = 1.5;
        g.beginPath(); g.arc(X(t), Y(e.manual), 4.5, 0, 2 * Math.PI); g.stroke();
      }
      g.fillStyle = ctxColor(e.context);
      g.beginPath(); g.arc(X(t), Y(e.rr), 4, 0, 2 * Math.PI); g.fill();
    }
    const used = CONTEXTS.filter(c => entries.some(e => e.context === c));
    $('trendLegend').replaceChildren(...used.map(c => el('span', null, el('i', { class: 'dot', style: `background:${ctxColor(c)}` }), c)),
      entries.some(e => Number.isFinite(e.manual)) ? el('span', null, el('i', { class: 'ring' }), '自己數的') : null);
  }
  function list() {
    $('entries').replaceChildren(...[...entries].reverse().map(e => el('li', { class: 'entry' },
      el('div', { class: 'e-main' },
        el('span', { class: 'e-time' }, when(e.t)),
        el('span', { class: 'e-ctx' }, el('i', { class: 'dot', style: `background:${ctxColor(e.context)}` }), e.context, e.posture ? ` · ${e.posture}` : ''),
        e.example ? el('span', { class: 'badge' }, '範例') : e.source === 'sim' ? el('span', { class: 'badge' }, '模擬') : null),
      el('div', { class: 'e-vals' },
        el('span', null, '呼吸 ', el('b', null, fmt(e.rr))),
        el('span', null, '心跳 ', el('b', null, e.hr ?? '--')),
        el('span', null, '品質 ', QLABEL[e.quality] || '--'),
        Number.isFinite(e.manual) ? el('span', null, `自己數 ${fmt(e.manual, 0)}(差 ${(e.rr - e.manual >= 0 ? '+' : '') + fmt(e.rr - e.manual)})`) : null),
      e.note ? el('div', { class: 'e-note' }, e.note) : null,
      el('button', { type: 'button', class: 'e-del', 'aria-label': `刪除 ${when(e.t)} 的紀錄`, onclick: () => remove(e.id) }, '刪除'))));
  }
  function remove(id) {
    entries = entries.filter(e => e.id !== id); store(entries); render();
  }
  function render() {
    if (onChange) onChange(entries);
    const has = entries.length > 0;
    $('logEmpty').hidden = has;
    $('logBody').hidden = !has;
    $('btnExport').disabled = !has; $('btnClearAll').disabled = !has;
    $('logCount').textContent = has ? `${entries.length} 筆` : '';
    if (!has) return;
    tiles(); contextTable(); list();
    requestAnimationFrame(drawTrend);
  }

  function exportCsv() {
    const cols = ['time', 'source', 'context', 'posture', 'rr_bpm', 'hr_bpm', 'quality', 'channel', 'self_count_rr', 'riiv', 'riav', 'rifv', 'note'];
    const q = v => { const s = v == null ? '' : String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
    const rows = entries.map(e => [e.t, e.example ? 'example' : e.source, e.context, e.posture, e.rr, e.hr, e.quality, e.channel,
      e.manual, e.parts?.riiv, e.parts?.riav, e.parts?.rifv, e.note].map(q).join(','));
    const blob = new Blob(['﻿' + [cols.join(','), ...rows].join('\n')], { type: 'text/csv;charset=utf-8' });
    const a = el('a', { href: URL.createObjectURL(blob), download: `ppg-log-${new Date().toISOString().slice(0, 10)}.csv` });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }
  function loadExamples() {
    const now = Date.now(), day = 86400e3;
    const ex = [
      [13, '起床後', '躺著', 14.2, 66, 14], [12.6, '工作或讀書', '坐著', 16.1, 74, null], [12, '運動後', '站著', 24.8, 112, 26],
      [11.5, '睡前', '躺著', 12.1, 62, 12], [10, '休息', '坐著', 14.6, 70, 14], [9.6, '壓力或焦慮', '坐著', 19.4, 88, 20],
      [8, '休息', '坐著', 15.1, 72, null], [6, '運動後', '站著', 22.7, 104, 22], [4, '休息', '躺著', 13.8, 68, 14],
      [3.5, '身體不適', '躺著', 21.3, 96, null], [2, '睡前', '躺著', 11.6, 60, 12], [1, '休息', '坐著', 14.9, 71, 16],
    ];
    const made = ex.map(([ago, c, p, rr, hr, man], i) => ({
      id: 'ex' + i, t: new Date(now - ago * day).toISOString(), source: 'example', example: true,
      rr, hr, quality: i % 5 === 3 ? 'fair' : 'good', channel: null, parts: null, context: c, posture: p,
      manual: man, note: c === '運動後' ? '爬完樓梯' : '',
    }));
    entries = [...entries.filter(e => !e.example), ...made].sort((a, b) => Date.parse(a.t) - Date.parse(b.t));
    store(entries); render();
  }

  let onChange = null;
  function init({ getSnapshot, onEntries }) {
    onChange = onEntries || null;
    $('btnSave').addEventListener('click', () => { const s = getSnapshot(); if (s) openForm(s); });
    $('btnCancelSave').addEventListener('click', closeForm);
    $('btnTimer').addEventListener('click', startTimer);
    $('saveForm').addEventListener('submit', submit);
    $('btnExport').addEventListener('click', exportCsv);
    $('btnExample').addEventListener('click', loadExamples);
    $('btnClearAll').addEventListener('click', () => { $('clearN').textContent = entries.length; $('clearConfirm').hidden = false; });
    $('btnClearNo').addEventListener('click', () => { $('clearConfirm').hidden = true; });
    $('btnClearYes').addEventListener('click', () => { entries = []; store(entries); $('clearConfirm').hidden = true; render(); });
    addEventListener('resize', () => entries.length && drawTrend());
    render();
  }
  window.PPGLog = { init };
})();
