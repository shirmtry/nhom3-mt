/* ===== Nhóm 3 · Dịch Việt/Pháp → Anh — giao diện ===== */
(function () {
  'use strict';
  const API = { status: '/api/status', warmup: '/api/warmup', translate: '/api/translate', results: '/api/results' };
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const f2 = (x) => (x == null || isNaN(x) ? '–' : Number(x).toFixed(2));
  const pct = (x) => (x == null ? '–' : Math.round(x * 1000) / 10 + '%');
  const sign = (x) => (x == null ? '–' : (x >= 0 ? '+' : '') + x.toFixed(2));
  const fmtMs = (n) => (n == null ? '' : n >= 1000 ? (n / 1000).toFixed(1).replace('.', ',') + ' s' : n + ' ms');
  const LANG = { vi: 'Tiếng Việt', fr: 'Français', en: 'English' };
  const MODE = { full: 'Có cổng', draft: 'Chỉ dịch nháp', always: 'Luôn tra kho' };
  const MODE_DESC = {
    full: 'Đề xuất. Câu dễ dùng luôn bản nháp; câu khó mới tra kho Pháp–Anh và chỉ dùng bằng chứng khi điểm rerank ≥ δ.',
    draft: 'Chỉ chạy bước ① ②: một lần dịch của Transformer, không tra kho.',
    always: 'Tra kho và dịch lại cho mọi câu (bỏ cổng) — dùng để so sánh vai trò của cổng.',
  };
  const SAMPLES = [
    { k: 'Câu thường', v: 'Tôi yêu gia đình của tôi.', src: 'auto' },
    { k: 'Mượn từ bằng chứng', v: 'Tôi xin lỗi, ông nói hoàn toàn đúng.', src: 'auto' },
    { k: 'Thành ngữ', v: 'Trung thành với con quỷ mà bạn biết.', src: 'auto' },
    { k: 'Bằng chứng làm hỏng', v: 'Hãy làm theo lời cha nhé.', src: 'auto' },
    { k: 'Từ ghép', v: 'Tôi là học sinh, em tôi là sinh viên.', src: 'auto' },
    { k: 'Tiếng Pháp', v: 'Je suis étudiant en intelligence artificielle.', src: 'auto' },
    { k: 'Nhiều câu', v: 'Tôi thích học máy học. Nó giúp tôi giải quyết nhiều bài toán thực tế.', src: 'auto' },
  ];

  const state = { source: 'auto', mode: 'full', last: null, resultsLoaded: false, poll: null };

  async function getJSON(url, opt) {
    const r = await fetch(url, opt);
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.error || r.statusText || 'Lỗi máy chủ');
    return d;
  }

  /* ---------------- tab ---------------- */
  document.querySelectorAll('.tab').forEach((b) => b.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((x) => { x.classList.toggle('on', x === b); x.setAttribute('aria-selected', x === b); });
    document.querySelectorAll('.panel').forEach((p) => p.classList.toggle('on', p.id === 'tab-' + b.dataset.tab));
    if (b.dataset.tab === 'results') loadResults();
  }));

  /* ---------------- trạng thái mô hình ---------------- */
  function setStatus(cls, text) {
    const el = $('status');
    el.className = 'status ' + cls;
    el.querySelector('.t').textContent = text;
    el.title = text;
  }

  async function checkStatus() {
    try {
      const s = await getJSON(API.status);
      if (!s.available) { setStatus('err', 'Chưa có mô hình — chạy python tai_mo_hinh.py'); return; }
      if (s.core_loaded && s.retrieval_loaded) { setStatus('ok', 'Sẵn sàng'); return; }
      const w = await getJSON(API.warmup, { method: 'POST' });
      setStatus('busy', 'Đang nạp mô hình… (lần đầu 1–2 phút)');
      clearInterval(state.poll);
      state.poll = setInterval(async () => {
        try {
          const st = await getJSON(API.status);
          if ((st.state || '').startsWith('lỗi')) { setStatus('err', st.state); clearInterval(state.poll); }
          else if (st.core_loaded && st.retrieval_loaded) { setStatus('ok', 'Sẵn sàng'); clearInterval(state.poll); }
          else setStatus('busy', (st.state || 'Đang nạp…').replace(/^./, (c) => c.toUpperCase()));
        } catch (e) { setStatus('err', 'Mất kết nối máy chủ'); clearInterval(state.poll); }
      }, 2000);
      void w;
    } catch (e) {
      setStatus('err', 'Không kết nối được máy chủ');
    }
  }

  /* ---------------- ô nhập ---------------- */
  const input = $('input');
  function bindSeg(id, key, after) {
    $(id).querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
      state[key] = b.dataset.v;
      $(id).querySelectorAll('button').forEach((x) => { x.classList.toggle('on', x === b); x.setAttribute('aria-pressed', x === b); });
      if (after) after();
    }));
  }
  function setSeg(id, key, v) {
    const b = $(id).querySelector(`button[data-v="${v}"]`);
    if (b) b.click();
    else state[key] = v;
  }
  bindSeg('srcSeg', 'source');
  bindSeg('modeSeg', 'mode', () => { $('modeDesc').textContent = MODE_DESC[state.mode]; });
  $('modeDesc').textContent = MODE_DESC.full;
  const updateCount = () => { $('count').textContent = `${input.value.length} / 2000`; };
  input.addEventListener('input', updateCount);
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); translate(); } });
  $('go').addEventListener('click', translate);
  $('clear').addEventListener('click', () => {
    input.value = ''; updateCount(); input.focus();
    showOutput('empty', 'Bản dịch sẽ hiện ở đây.');
    $('outMeta').textContent = ''; $('copy').disabled = true; $('pipeCard').hidden = true;
  });
  $('copy').addEventListener('click', async () => {
    const t = state.last ? state.last.translation : '';
    try { await navigator.clipboard.writeText(t); flash('Đã sao chép'); }
    catch (e) {
      const r = document.createRange(); r.selectNodeContents($('output'));
      const s = window.getSelection(); s.removeAllRanges(); s.addRange(r); flash('Bấm Ctrl + C để sao chép');
    }
  });
  function flash(msg) {
    const b = $('copy'), old = 'Sao chép';
    b.textContent = msg; setTimeout(() => { b.textContent = old; }, 1400);
  }

  $('samples').innerHTML = SAMPLES.map((s, i) =>
    `<button type="button" class="sample" data-i="${i}"><span class="k">${esc(s.k)}</span><span class="v">${esc(s.v)}</span></button>`).join('');
  $('samples').querySelectorAll('.sample').forEach((b) => b.addEventListener('click', () => {
    const s = SAMPLES[+b.dataset.i];
    input.value = s.v; updateCount(); setSeg('srcSeg', 'source', s.src);
    translate();
  }));

  function showOutput(cls, text) {
    const o = $('output');
    o.className = 'output' + (cls ? ' ' + cls : '');
    o.textContent = text;
  }

  /* ---------------- dịch ---------------- */
  async function translate() {
    const text = input.value.trim();
    if (!text) { input.focus(); return; }
    const btn = $('go');
    btn.disabled = true; btn.textContent = 'Đang dịch…';
    showOutput('loading', 'Mô hình đang dịch… (câu khó có tra kho sẽ lâu hơn)');
    try {
      const d = await getJSON(API.translate, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, source: state.source, mode: state.mode }),
      });
      state.last = d;
      showOutput('', d.translation);
      const det = d.detected ? ` (tự phát hiện)` : '';
      $('outMeta').textContent = `${LANG[d.source] || d.source}${det} → English · ${fmtMs(d.elapsed_ms)}`;
      $('copy').disabled = false;
      renderPipeline(d);
      if (window.innerWidth <= 900) document.querySelector('.o-out').scrollIntoView({ behavior: 'smooth', block: 'start' });
      addHistory({ text: d.original, out: d.translation, src: d.source, mode: d.mode, t: Date.now() });
      if (!$('status').classList.contains('ok')) checkStatus();
    } catch (e) {
      showOutput('error', e.message);
      $('outMeta').textContent = ''; $('copy').disabled = true; $('pipeCard').hidden = true;
    } finally {
      btn.disabled = false; btn.textContent = 'Dịch sang tiếng Anh';
    }
  }

  /* ---------------- thẻ "Mô hình đã làm gì" ---------------- */
  const step = (cls, num, name, ms, html) => `<div class="step ${cls}"><div class="num">${num}</div><div class="body">
    <div class="head"><span class="name">${name}</span><span class="ms">${fmtMs(ms)}</span></div>${html}</div></div>`;
  const off = (num, name, why) => step('off', num, name, null, `<div class="sub">không chạy — ${why}</div>`);

  function gauge(label, v, th, thName, badWhenAbove) {
    const has = th != null && th <= 1;
    const bad = has && (badWhenAbove ? v > th : v < th);
    const mark = has ? `<div class="mark" style="left:${Math.min(100, th * 100)}%"></div>` : '';
    return `<div class="gauge ${bad ? 'bad' : 'ok'}"><div class="lbl"><span>${label}</span><span>${has ? thName + ' = ' + f2(th) : thName + ': không dùng'}</span></div>
      <div class="val">${f2(v)}</div><div class="bar"><div class="fill" style="width:${Math.min(100, (v || 0) * 100)}%"></div>${mark}</div></div>`;
  }

  function sentenceHTML(s, i, n, mode) {
    const th = s.thresholds || {}, T = s.timings_ms || {};
    const why = [s.reasons && s.reasons.C_low ? 'độ tự tin thấp' : null, s.reasons && s.reasons.R_high ? 'nhiều từ hiếm' : null].filter(Boolean).join(' + ');
    let verdict;
    if (!s.hard) {
      verdict = mode === 'draft'
        ? `<div class="verdict easy"><span class="ic">📝</span>Chế độ chỉ dịch nháp: dừng ở bước ②</div>`
        : `<div class="verdict easy"><span class="ic">✅</span>Câu dễ → dùng luôn bản nháp, không tra kho Pháp–Anh</div>`;
    } else if (s.evidence_used) {
      verdict = `<div class="verdict hard"><span class="ic">🔎</span>${mode === 'always' ? 'Chế độ luôn tra kho' : 'Câu khó (' + esc(why) + ')'} → tra kho Pháp–Anh, dịch lại có bằng chứng</div>`;
    } else {
      verdict = `<div class="verdict weak"><span class="ic">⚠️</span>Câu khó nhưng ${s.evidence ? `bằng chứng chưa đủ tốt (rerank ${f2(s.ev_score)} &lt; δ = ${f2(th.delta)})` : 'không tìm được bằng chứng'} → giữ bản nháp</div>`;
    }
    const pieces = (s.pieces || []).slice(0, 18).map((p, j) => `<span class="chip${j === 0 ? ' tag' : ''}">${esc(p)}</span>`).join('')
      + ((s.n_pieces || 0) > 18 ? '<span class="chip">…</span>' : '');
    const segW = (s.words || []).map((w) => `<span class="chip word${/\s/.test(w) ? ' multi' : ''}">${esc(w)}</span>`).join('');
    const nMulti = (s.words || []).filter((w) => /\s/.test(w)).length;
    const s1 = step('base', 1, 'Tiền xử lý · chuẩn hóa, tách từ, tách mảnh từ, gắn mã ngôn ngữ', T.preprocess,
      (segW ? `<div class="sub lvl"><b>Tách từ</b> (${esc(s.seg_tool)}) → dùng cho độ hiếm R ở bước ③${nMulti ? ` · ${nMulti} từ ghép` : ''}</div><div class="chips">${segW}</div>` : '')
      + `<div class="sub lvl"><b>Mảnh từ</b> (SentencePiece của mBART-50) → đầu vào của Transformer</div><div class="chips">${pieces}</div>`
      + `<div class="sub">${s.n_pieces} mảnh · mã nguồn <b>[${esc(s.lang_code)}]</b>, đầu ra luôn bắt đầu bằng <b>[en_XX]</b></div>`);
    const s2 = step('core', 2, 'Multilingual Transformer · dịch nháp ŷ', T.draft,
      `<div class="txt en">${esc(s.draft)}</div><div class="sub">mBART-50 + LoRA, một bộ tham số học chung Việt–Anh và Pháp–Anh</div>`);
    const gateResult = mode === 'draft' ? 'Chế độ chỉ dịch nháp: bỏ qua cổng'
      : mode === 'always' ? 'Chế độ luôn tra kho: bỏ qua cổng'
      : s.hard ? '→ <b>Khó</b> (C &lt; τ hoặc R &gt; ρ): tra kho' : '→ <b>Dễ</b> (C ≥ τ và R ≤ ρ): dùng luôn ŷ';
    const rare = (s.rare_words || []).length
      ? `<div class="sub">Từ ít gặp trong dữ liệu huấn luyện: ${s.rare_words.map((w) => `<span class="chip rare">${esc(w)}</span>`).join(' ')}</div>` : '';
    const s3 = step('ctrl', 3, 'Cổng quyết định', null,
      `<div class="gauges">${gauge('Độ tự tin C', s.C, th.tau, 'τ', false)}${gauge('Độ hiếm R', s.R, th.rho, 'ρ', true)}</div>${rare}<div class="sub" style="margin-top:6px">${gateResult}</div>`);
    const reason = !s.hard ? (mode === 'draft' ? 'chế độ chỉ dịch nháp' : 'câu dễ') : 'bằng chứng chưa đủ tốt';
    let s4, s5;
    if (s.hard && (s.candidates || []).length) {
      const rows = s.candidates.map((c, j) => `<tr class="${c.chosen ? 'chosen' : ''}"><td>${j + 1}</td>
        <td><div class="fr">FR · ${esc(c.fr)}</div><div>EN · ${esc(c.en)}${c.chosen ? '<span class="pick">e*</span>' : ''}</div></td>
        <td class="r">${f2(c.cosine)}</td><td class="r"><b>${f2(c.rerank)}</b></td></tr>`).join('');
      s4 = step('aug', 4, 'Tra kho Pháp–Anh · LaBSE lấy Top-5', T.retrieve,
        `<div class="scroll"><table class="tbl"><thead><tr><th>#</th><th>Câu Pháp (khóa tìm) → câu Anh đi kèm</th><th class="r">cosine</th><th class="r">⑤ rerank</th></tr></thead><tbody>${rows}</tbody></table></div>
         <div class="sub">${s.lang === 'vi' ? 'Câu Việt tìm xuyên ngôn ngữ sang câu Pháp' : 'Câu Pháp tìm câu Pháp giống nhất'} · không dịch qua tiếng Pháp</div>`);
      s5 = step('aug', 5, 'Chọn bằng chứng · reranker đọc cả cặp câu', T.rerank,
        `<div class="txt">e* = ${esc(s.evidence)}</div>
         <div class="sub">điểm rerank ${f2(s.ev_score)} ${s.evidence_used ? (mode === 'always' ? '· chế độ luôn tra kho' : '≥ δ = ' + f2(th.delta) + ' → dùng') : '&lt; δ = ' + f2(th.delta) + ' → bỏ qua, giữ bản nháp'}</div>`);
    } else {
      s4 = off(4, 'Tra kho Pháp–Anh', reason);
      s5 = off(5, 'Chọn bằng chứng', reason);
    }
    const s6 = s.refined
      ? step('aug', 6, 'Dịch lại · cùng Transformer, đầu vào x + e*', T.refine,
          `<div class="txt en">${esc(s.refined)}</div><div class="sub">đầu vào: [${esc(s.lang_code)}] x &lt;/s&gt; e* &lt;/s&gt;</div>`)
      : off(6, 'Dịch lại với x + e*', reason);
    const title = n > 1 ? `<div class="sent-title">Câu ${i + 1}/${n}: ${esc(s.src)}</div>` : '';
    return `<div class="sent">${title}${verdict}<div class="steps">${s1}${s2}${s3}${s4}${s5}${s6}</div></div>`;
  }

  function renderPipeline(d) {
    $('pipeMode').textContent = MODE[d.mode] || d.mode;
    $('pipeTime').textContent = fmtMs(d.elapsed_ms);
    const n = d.sentences.length;
    $('pipe').innerHTML = d.sentences.map((s, i) => sentenceHTML(s, i, n, d.mode)).join('');
    $('pipeCard').hidden = false;
  }

  /* ---------------- lịch sử (chỉ lưu trên trình duyệt này) ---------------- */
  const HKEY = 'nhom3-history';
  function readHist() { try { return JSON.parse(localStorage.getItem(HKEY) || '[]'); } catch (e) { return []; } }
  function writeHist(h) { try { localStorage.setItem(HKEY, JSON.stringify(h.slice(0, 20))); } catch (e) { /* bỏ qua */ } }
  let memHist = readHist();
  function addHistory(item) {
    memHist = [item, ...memHist.filter((x) => x.text !== item.text)].slice(0, 20);
    writeHist(memHist); renderHistory();
  }
  function renderHistory() {
    const ul = $('history');
    if (!memHist.length) { ul.innerHTML = '<li class="muted">Chưa có câu nào.</li>'; return; }
    ul.innerHTML = memHist.map((h, i) => `<li data-i="${i}"><div class="h-meta">${esc(LANG[h.src] || h.src)} → English · ${esc(MODE[h.mode] || h.mode)}</div>
      <div class="h-src">${esc(h.text)}</div><div class="h-out">${esc(h.out)}</div></li>`).join('');
    ul.querySelectorAll('li[data-i]').forEach((li) => li.addEventListener('click', () => {
      const h = memHist[+li.dataset.i];
      input.value = h.text; updateCount(); setSeg('modeSeg', 'mode', h.mode || 'full'); input.focus();
    }));
  }
  $('clearHist').addEventListener('click', () => { memHist = []; writeHist(memHist); renderHistory(); });
  renderHistory();

  /* ---------------- tab "Kết quả thử nghiệm" ---------------- */
  const DESC = {
    'A0/draft': 'Transformer nhỏ học từ đầu · chỉ Việt–Anh',
    'A/draft': 'mBART-50 + LoRA · chỉ Việt–Anh',
    'B/draft': '+ Pháp–Anh học chung',
    'C/draft': 'C · chỉ dịch nháp',
    'C/full': 'C · có cổng + tra kho (đề xuất)',
    'C/always': 'C · luôn tra kho',
  };
  const card = (title, html) => `<div class="card"><div class="card-title">${title}</div>${html}</div>`;

  function resultsHTML(d) {
    if (!d.available) {
      return card('Kết quả thử nghiệm', '<div class="empty-box"><b>Chưa có kết quả</b>Chạy <code>python tai_mo_hinh.py</code> để tải mô hình kèm kết quả.</div>');
    }
    const rows = [];
    d.experiments.forEach((ex) => ['draft', 'full', 'always'].forEach((m) => {
      const vi = ex.results['vi/' + m];
      if (vi) rows.push({ run: ex.run, mode: m, vi, fr: ex.results['fr/' + m] || null, gate: ex.results['vi/gate'] });
    }));
    const get = (run, mode) => rows.find((r) => r.run === run && r.mode === mode);
    const best = rows.reduce((a, r) => (!a || r.vi.BLEU > a.vi.BLEU ? r : a), null);
    const maxB = Math.max(...rows.map((r) => r.vi.BLEU), 1);
    const A0 = get('A0', 'draft'), A = get('A', 'draft'), B = get('B', 'draft'), CD = get('C', 'draft'), CF = get('C', 'full'), CA = get('C', 'always');

    const tiles = [];
    if (A0 && A) tiles.push({ v: `${f2(A0.vi.BLEU)} → ${f2(A.vi.BLEU)}`, l: 'Tiền huấn luyện', s: 'BLEU Việt→Anh, A0 → A', hl: true });
    if (A && B) tiles.push({ v: sign(B.vi.BLEU - A.vi.BLEU), l: 'Học chung Pháp–Anh', s: 'BLEU B so với A' });
    if (CD && CF) tiles.push({ v: sign(CF.vi.BLEU - CD.vi.BLEU), l: 'Cổng + bằng chứng', s: 'BLEU C có cổng so với chỉ nháp' });
    if (CA && CF) tiles.push({ v: sign(CF.vi.BLEU - CA.vi.BLEU), l: 'Vai trò của cổng', s: 'có cổng so với luôn tra kho' });
    if (CF && CF.vi.evidence_frac != null) tiles.push({ v: pct(CF.vi.evidence_frac), l: 'Câu dùng bằng chứng', s: 'trên test Việt–Anh' });
    if (d.retrieval && d.retrieval.n) tiles.push({ v: pct(d.retrieval['Recall@5_LaBSE']), l: 'Recall@5', s: 'truy xuất LaBSE' });
    const tileHTML = `<div class="tiles">${tiles.map((t) => `<div class="tile${t.hl ? ' hl' : ''}"><div class="v">${t.v}</div><div class="l">${t.l}</div><div class="s">${t.s}</div></div>`).join('')}</div>`;

    const trs = rows.map((r) => {
      const key = r.run + '/' + r.mode, isBest = r === best;
      return `<tr class="${isBest ? 'best' : ''} ${key === 'C/full' ? 'proposed' : ''}">
        <td><b>${esc(r.run)}</b>${r.mode !== 'draft' ? ' · ' + (r.mode === 'full' ? 'cổng' : 'luôn') : ''}</td>
        <td class="muted">${esc(DESC[key] || key)}</td>
        <td><div class="hbar${isBest ? ' best' : ''}"><div class="track"><div class="fill" style="width:${(r.vi.BLEU / maxB) * 100}%"></div></div><span class="v">${f2(r.vi.BLEU)}</span></div></td>
        <td class="r">${f2(r.vi.chrF)}</td><td class="r">${r.fr ? f2(r.fr.BLEU) : '–'}</td><td class="r">${r.fr ? f2(r.fr.chrF) : '–'}</td></tr>`;
    }).join('');
    const table = `<div class="scroll"><table class="tbl"><thead><tr><th>Mô hình</th><th>Cấu hình</th><th>BLEU Việt → Anh</th><th class="r">chrF</th><th class="r">BLEU Pháp → Anh</th><th class="r">chrF</th></tr></thead><tbody>${trs}</tbody></table></div>
      <p class="note">Tập test: 1.000 câu Việt–Anh (PhoMT) và 500 câu Pháp–Anh. Hàng xanh: BLEU Việt → Anh cao nhất. Chênh lệch dưới khoảng 1 BLEU có thể nằm trong sai số.</p>`;

    let gateHTML = '';
    const cal = d.calibration && d.calibration.thresholds;
    if (cal) {
      const order = (l) => (l === 'vi' ? 0 : 1);
      const tr = Object.entries(cal).sort((x, y) => order(x[0]) - order(y[0])).map(([l, t]) => `<tr><td><b>${l === 'vi' ? 'Tiếng Việt' : 'Tiếng Pháp'}</b></td>
        <td class="r">${f2(t.tau)}</td><td class="r">${t.rho > 1 ? '–' : f2(t.rho)}</td><td class="r">${t.delta > 1 ? '–' : f2(t.delta)}</td>
        <td class="r">${f2(t.draft_only_util)}</td><td class="r"><b>${f2(t.dev_util)}</b></td><td class="r">${f2(t.always_util)}</td>
        <td class="r">${pct(t.hard_frac)}</td><td class="r">${pct(t.evidence_frac)}</td></tr>`).join('');
      const diag = d.calibration.diagnostics || {};
      const dg = Object.entries(diag).sort((x, y) => order(x[0]) - order(y[0])).map(([l, x]) => `${l === 'vi' ? 'Việt' : 'Pháp'}: AUROC(C) = ${f2(x.AUROC_C)}, AUROC(R) = ${f2(x.AUROC_R)}`).join(' · ');
      gateHTML = card('Cổng · ngưỡng chọn trên tập dev', `<div class="scroll"><table class="tbl"><thead><tr><th></th><th class="r nocase">τ</th><th class="r nocase">ρ</th><th class="r nocase">δ</th>
        <th class="r">chỉ nháp</th><th class="r">có cổng</th><th class="r">luôn tra</th><th class="r">% khó</th><th class="r">% bằng chứng</th></tr></thead><tbody>${tr}</tbody></table></div>
        <p class="note">Ba cột giữa: chrF câu trung bình trên dev. ${esc(dg)} (AUROC &gt; 0,5 nghĩa là chỉ số giúp nhận ra bản nháp kém).</p>`);
    }

    let retrHTML = '';
    if (d.retrieval && d.retrieval.n) {
      const r = d.retrieval;
      retrHTML = card(`Truy xuất trong kho Pháp–Anh · ${Number(r.n).toLocaleString('vi-VN')} câu kiểm tra`, `<div class="tiles">
        <div class="tile"><div class="v">${pct(r['Recall@1_LaBSE'])}</div><div class="l">Recall@1</div><div class="s">chỉ LaBSE</div></div>
        <div class="tile"><div class="v">${pct(r['Recall@5_LaBSE'])}</div><div class="l">Recall@5</div><div class="s">LaBSE Top-5</div></div>
        <div class="tile hl"><div class="v">${pct(r['Recall@1_after_rerank'])}</div><div class="l">Recall@1</div><div class="s">sau rerank</div></div></div>`);
    }

    const exs = d.examples || [];
    const nLeak = exs.filter((e) => e.leak).length;
    const exHTML = exs.length ? card('Ví dụ câu khó trên tập test', exs.map((e) => `<div class="ex">
      <div class="tags">${e.leak ? '<span class="tagx leak">bằng chứng ≈ đáp án</span>' : ''}<span class="tagx ${e.gain >= 0 ? 'up' : 'down'}">chrF ${e.gain >= 0 ? '+' : ''}${Number(e.gain).toFixed(1)}</span></div>
      <div class="r"><span class="k">Câu nguồn</span><span>${esc(e.src)}</span></div>
      <div class="r"><span class="k">Bản nháp</span><span class="draft">${esc(e.draft)}</span></div>
      <div class="r"><span class="k">Bằng chứng</span><span class="ev">${esc(e.evidence)} <span class="fr">(rerank ${f2(e.ev_score)})</span></span></div>
      <div class="r"><span class="k">Dịch lại</span><span class="out">${esc(e.output)}</span></div>
      <div class="r"><span class="k">Đáp án</span><span>${esc(e.ref)}</span></div></div>`).join('')
      + `<p class="note">Gồm các câu được sửa tốt nhất và các câu bị sửa hỏng.${nLeak ? ` ${nLeak} câu có bằng chứng gần trùng đáp án: PhoMT và TED2020 cùng lấy từ các bài nói TED, nên kho Pháp–Anh có thể chứa câu Anh của chính câu test. Phần cải thiện của C vì vậy có thể bị đánh giá cao hơn thực tế.` : ''}</p>`) : '';

    return `<div class="results-stack">${card('Kết quả trên tập test', tileHTML + table)}${gateHTML}${retrHTML}${exHTML}</div>`;
  }

  async function loadResults() {
    if (state.resultsLoaded) return;
    try {
      const d = await getJSON(API.results);
      $('results').innerHTML = resultsHTML(d);
      state.resultsLoaded = d.available;
    } catch (e) {
      $('results').innerHTML = card('Kết quả thử nghiệm', `<div class="empty-box"><b>Không tải được kết quả</b>${esc(e.message)}</div>`);
    }
  }

  checkStatus();
})();
