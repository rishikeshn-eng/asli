/* asli site: in-browser detector (same preprocessing as asli/detect.py) + results. */
(async function () {
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const pct = (x, d = 1) => (x == null ? "–" : (x * 100).toFixed(d) + "%");
  const SR = 16000, WIN = 64000, HOP = 32000;
  let sess = null, META = null;

  const R = await fetch("data/report.json").then((r) => r.json()).catch(() => null);
  if (R) renderResults(R);

  try {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    [sess, META] = await Promise.all([ort.InferenceSession.create("model/asli.onnx", { executionProviders: ["wasm"] }),
      fetch("model/asli.json").then((r) => r.json())]);
    $("#state").textContent = `model ready · ${META.params.toLocaleString()} parameters · runs locally`;
  } catch (e) { $("#state").textContent = "model failed to load: " + e.message; return; }

  // ---------------------------------------------------------------- audio pipeline
  async function decode(arrayBuf) {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const buf = await ctx.decodeAudioData(arrayBuf);
    ctx.close();
    const off = new OfflineAudioContext(1, Math.ceil(buf.duration * SR), SR);
    const src = off.createBufferSource();
    src.buffer = buf; src.connect(off.destination); src.start();
    return (await off.startRendering()).getChannelData(0).slice();
  }
  function trim(x, topDb = 35) {
    const hop = 160, win = 400;
    if (x.length < win * 2) return x;
    const n = Math.floor((x.length - win) / hop) + 1, rms = new Float32Array(n);
    let mx = 0;
    for (let i = 0; i < n; i++) { let s = 0; for (let k = 0; k < win; k++) { const v = x[i * hop + k]; s += v * v; } rms[i] = Math.sqrt(s / win + 1e-12); if (rms[i] > mx) mx = rms[i]; }
    let a = -1, b = -1;
    for (let i = 0; i < n; i++) if (20 * Math.log10(rms[i] / mx) > -topDb) { if (a < 0) a = i; b = i; }
    if (a < 0) return x;
    return x.slice(Math.max(0, a * hop - 1600), Math.min(x.length, b * hop + win + 1600));
  }
  function normalise(x, db = -24) {
    let s = 0; for (const v of x) s += v * v;
    const g = Math.pow(10, db / 20) / Math.sqrt(s / x.length + 1e-12);
    const y = x.map((v) => v * g);
    let pk = 0; for (const v of y) pk = Math.max(pk, Math.abs(v));
    return pk > 0.99 ? y.map((v) => (v / pk) * 0.99) : y;
  }
  function windows(x) {
    if (x.length < WIN) { const w = new Float32Array(WIN); for (let i = 0; i < WIN; i++) w[i] = x[i % x.length]; return [[w], [0]]; }
    const starts = [];
    for (let s = 0; s + WIN <= x.length; s += HOP) starts.push(s);
    if (starts[starts.length - 1] + WIN < x.length) starts.push(x.length - WIN);
    return [starts.map((s) => x.slice(s, s + WIN)), starts.map((s) => s / SR)];
  }

  async function analyse(arrayBuf, label) {
    $("#state").textContent = "analysing " + label + "…";
    const raw = await decode(arrayBuf);
    const x = normalise(trim(raw));
    const [ws, starts] = windows(x);
    const B = ws.length, data = new Float32Array(B * WIN);
    ws.forEach((w, i) => data.set(w, i * WIN));
    const out = await sess.run({ wav: new ort.Tensor("float32", data, [B, WIN]) });
    const p = Array.from(out.p_synthetic.data), mel = out.logmel;
    const mean = p.reduce((a, b) => a + b, 0) / p.length;
    draw(mel, p, x.length / SR, starts);
    const thr = META.threshold;
    const v = Math.abs(mean - thr) < 0.1 ? "uncertain" : mean >= thr ? "synthetic" : "real";
    const T = { real: "No synthetic-speech artefacts found", synthetic: "Likely AI-generated speech", uncertain: "Unclear: treat with caution" };
    $("#verdict").className = "verdict " + v;
    $("#verdict").innerHTML = `<div class="v-title">${T[v]}</div><div class="v-sub">${esc(label)} · ${(raw.length / SR).toFixed(1)} s · ${B} window${B > 1 ? "s" : ""} · highest window ${pct(Math.max(...p), 0)}${v !== "synthetic" ? '<br><b>This does not rule out a voice clone:</b> unseen cloning models (e.g. XTTS-v2) are mostly missed. See results below.' : ""}</div>`;
    $("#gval").textContent = pct(mean, 0);
    const arc = $("#garc");
    arc.style.strokeDashoffset = 283 * (1 - mean);
    arc.style.stroke = v === "real" ? "var(--real)" : v === "synthetic" ? "var(--fake)" : "var(--sun)";
    $("#state").textContent = "done · ran locally";
  }

  // ---------------------------------------------------------------- drawing
  const MAGMA = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 253, 191]];
  const cmap = (t) => { t = Math.max(0, Math.min(1, t)) * (MAGMA.length - 1); const i = Math.floor(t), f = t - i, a = MAGMA[i], b = MAGMA[Math.min(i + 1, MAGMA.length - 1)];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]; };
  function draw(mel, p, dur, starts) {
    const [B, F, T] = mel.dims, d = mel.data;
    const cols = [];
    for (let b = 0; b < B; b++) {
      const take = b === B - 1 ? T : Math.min(T, Math.round(((starts[b + 1] ?? starts[b] + 2) - starts[b]) * 100));
      for (let t = 0; t < take; t++) { const c = new Float32Array(F); for (let f = 0; f < F; f++) c[f] = d[(b * F + f) * T + t]; cols.push([c, p[b]]); }
    }
    const all = cols.flatMap(([c]) => Array.from(c)).sort((a, b) => a - b);
    const lo = all[Math.floor(all.length * 0.05)], hi = all[Math.floor(all.length * 0.995)];
    const cv = $("#spec"), g = cv.getContext("2d"), img = g.createImageData(cols.length, F);
    cols.forEach(([c], x) => { for (let f = 0; f < F; f++) { const [r, gg, bb] = cmap((c[f] - lo) / (hi - lo)); const i = ((F - 1 - f) * cols.length + x) * 4;
      img.data[i] = r; img.data[i + 1] = gg; img.data[i + 2] = bb; img.data[i + 3] = 255; } });
    const tmp = document.createElement("canvas"); tmp.width = cols.length; tmp.height = F; tmp.getContext("2d").putImageData(img, 0, 0);
    g.imageSmoothingEnabled = true; g.clearRect(0, 0, cv.width, cv.height); g.drawImage(tmp, 0, 0, cv.width, cv.height);
    const tl = $("#timeline"), h = tl.getContext("2d");
    h.clearRect(0, 0, tl.width, tl.height);
    cols.forEach(([, pv], x) => { const w = tl.width / cols.length; const thr = META.threshold;
      h.fillStyle = pv >= thr ? `rgba(255,91,58,${0.35 + 0.65 * pv})` : `rgba(61,214,168,${0.35 + 0.65 * (1 - pv)})`; h.fillRect(x * w, 0, w + 0.6, tl.height); });
    const ticks = 6;
    $("#axis").innerHTML = Array.from({ length: ticks + 1 }, (_, i) => `<span>${((dur * i) / ticks).toFixed(1)}s</span>`).join("");
  }

  // ---------------------------------------------------------------- inputs
  $("#file").addEventListener("change", async (e) => { const f = e.target.files[0]; if (!f) return;
    $("#player").src = URL.createObjectURL(f); analyse(await f.arrayBuffer(), f.name); });
  $("#rec").addEventListener("click", async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream), chunks = [];
      mr.ondataavailable = (e) => chunks.push(e.data);
      mr.onstop = async () => { stream.getTracks().forEach((t) => t.stop()); const blob = new Blob(chunks, { type: mr.mimeType });
        $("#player").src = URL.createObjectURL(blob); $("#rec").textContent = "● Record 6 s"; $("#rec").classList.remove("rec");
        analyse(await blob.arrayBuffer(), "your recording"); };
      mr.start(); $("#rec").textContent = "recording…"; $("#rec").classList.add("rec");
      setTimeout(() => mr.stop(), 6000);
    } catch (e) { $("#state").textContent = "microphone unavailable: " + e.message; }
  });
  const samples = await fetch("samples/samples.json").then((r) => r.json()).catch(() => []);
  samples.forEach((s, i) => {
    const b = document.createElement("button");
    b.textContent = s.label; b.className = s.real ? "r" : "f"; b.title = s.real ? "real speech (FLEURS)" : "synthetic (MMS-TTS)";
    b.onclick = async () => { document.querySelectorAll(".samples button").forEach((x) => x.classList.remove("on")); b.classList.add("on");
      $("#player").src = "samples/" + s.file; analyse(await fetch("samples/" + s.file).then((r) => r.arrayBuffer()), s.label + (s.real ? " · real" : " · synthetic")); };
    $("#samples").appendChild(b);
  });
  if (samples.length) document.querySelector(".samples button").click();

  // ---------------------------------------------------------------- results
  function renderResults(R) {
    $("#kpis").innerHTML = R.kpis.map((k) => `<div class="kpi"><b>${k.v}</b><span>${k.l}</span></div>`).join("");
    $("#evalNote").innerHTML = R.eval_note;
    const bar = (label, v, max, color, extra = "") => `<div class="hb"><span>${label}${extra}</span><div class="tr"><span style="width:${Math.min(100, (v / max) * 100)}%;background:${color}"></span></div><span class="v">${pct(v)}</span></div>`;
    const CH = { clean: "Clean", phone: "Phone line (8 kHz)", opus: "WhatsApp-like Opus", mp3: "MP3 32 kbps", noise: "Noise, 10 dB SNR" };
    const maxC = Math.max(0.1, ...Object.values(R.channels).map((c) => c.eer));
    $("#chanChart").innerHTML = Object.entries(R.channels).map(([k, c]) => bar(CH[k], c.eer, maxC, "var(--sun)")).join("");
    const maxG = Math.max(0.1, ...Object.values(R.generators).map((g) => g.eer));
    $("#genChart").innerHTML = Object.entries(R.generators).map(([k, g]) =>
      bar(g.label, g.eer, maxG, g.seen ? "var(--real)" : "var(--fake)", g.seen ? " <small>seen</small>" : " <small>unseen</small>")).join("");
    $("#langSub").textContent = R.lang_sub;
    $("#langGrid").innerHTML = Object.values(R.languages).map((l) => `<div class="lang ${l.unseen ? "unseen" : ""}"><div class="n">${esc(l.name)}</div>
      <div class="s">${l.unseen ? "never trained on" : l.n + " clips"}</div><b>${l.eer != null ? pct(l.eer) : pct(l.flag_rate)}</b><div class="s">${l.eer != null ? "EER" : "flagged"}</div></div>`).join("");
    $("#baseTbl").innerHTML = `<table><tr><th>Detector</th><th>Params</th><th>Clean</th><th>Phone</th><th>Opus</th><th>Unseen engines</th><th>Voice clones</th></tr>` +
      R.baselines.map((b) => `<tr class="${b.us ? "us" : ""}"><td>${esc(b.name)}</td><td class="n">${esc(b.params)}</td><td class="n">${pct(b.clean)}</td><td class="n">${pct(b.phone)}</td><td class="n">${pct(b.opus)}</td><td class="n">${pct(b.unseen)}</td><td class="n">${pct(b.clones)}</td></tr>`).join("") + "</table>";
    const S = R.shortcut;
    $("#shortcut").innerHTML = `<div><div class="p-t">Shortcut test</div><div class="p-s">Does it just detect MP3 artefacts? Real test speech passed through the exact encoder one TTS engine uses.</div></div>
      <div><b style="color:var(--real)">${pct(S.real_clean_fpr)}</b><span class="p-s">false alarms on real speech</span></div>
      <div><b style="color:${S.real_through_edge_mp3_fpr > 2 * S.real_clean_fpr + 0.03 ? "var(--fake)" : "var(--real)"}">${pct(S.real_through_edge_mp3_fpr)}</b><span class="p-s">false alarms on the same speech after MP3</span></div>`;
    $("#dataStats").innerHTML = R.data.map((d) => `<div class="ds"><b>${d.v}</b><span>${d.l}</span></div>`).join("");
  }
})();
