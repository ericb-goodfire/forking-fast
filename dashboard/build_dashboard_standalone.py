"""Build the fully self-contained reduced-run dashboard.

One HTML file, zero network or file dependencies: plotly.js inlined from the
Python plotly package, both datasets embedded as application/json blocks,
no fetch anywhere. Works from file:// with no server. Drops the legacy
5-question track (its rows exist in the 50-question set).

Usage: python dashboard/build_dashboard_standalone.py \
    --llama <dashboard_llama.json> --deepseek <dashboard_deepseek.json> \
    --tokens tokens_meta.json --changepoints cps_meta.json \
    --out dashboard_standalone.html
"""
import argparse
import json
import os

import plotly

TRACK_META = {
    "llama": "Llama-3-8B-Instruct — every-token grid",
    "deepseek": "DeepSeek-R1-Distill-Llama-8B — sentence grid",
}

HTML_TOP = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reduced-run explorer — 100 questions, two models (standalone)</title>
<style>
  :root { --ink:#1D272A; --muted:#6b6459; --line:#d8d2c7; --ember:#C4650D; }
  body { margin:0; padding:24px 32px 48px; color:var(--ink);
         font:14px/1.5 'Suisse Intl',-apple-system,BlinkMacSystemFont,Arial,sans-serif;
         background:#faf8f4; }
  h1 { font-family:Georgia,'Times New Roman',serif; font-weight:600;
       font-size:24px; margin:0 0 6px; }
  p.lead { max-width:72ch; color:var(--muted); margin:0 0 18px; }
  .controls { display:flex; gap:18px; flex-wrap:wrap; align-items:flex-end;
              margin-bottom:6px; }
  .ctrl label { display:block; font-size:12px; color:var(--muted);
                margin-bottom:3px; }
  select { font:13px inherit; padding:5px 8px; border:1px solid var(--line);
           border-radius:4px; background:#fff; color:var(--ink);
           min-width:120px; max-width:520px; }
  .radios { display:inline-flex; gap:14px; align-items:center;
            border:1px solid var(--line); border-radius:4px;
            background:#fff; padding:5px 10px; }
  .radios label { font-size:13px; color:var(--ink); white-space:nowrap; }
  .radios input { margin:0 5px 0 0; vertical-align:-1px; }
  .chips { margin:10px 0 2px; }
  .chip { display:inline-block; border:1px solid var(--line);
          border-radius:12px; padding:2px 10px; margin:0 6px 6px 0;
          font-size:12px; background:#fff; }
  .chip b { color:var(--ember); }
  #fig { width:100%; }
  .tokhead { margin:22px 0 6px; font-size:12px; color:var(--muted); }
  .tokpanel { background:#fff; border:1px solid var(--line);
              border-radius:6px; padding:14px 16px;
              white-space:pre-wrap; overflow-wrap:anywhere;
              font:13px/1.75 'Suisse Intl',-apple-system,Arial,sans-serif; }
  .tok:hover { background:#f3e2cd; box-shadow:0 0 0 1px var(--ember);
               border-radius:2px; }
  #tokTip { position:fixed; display:none; pointer-events:none;
            background:var(--ink); color:#faf8f4; padding:3px 10px;
            border-radius:999px; font-size:12px; z-index:10;
            white-space:pre; max-width:520px; overflow:hidden;
            text-overflow:ellipsis; }
</style>
</head>
<body>
<h1>Reduced-run explorer</h1>
<p class="lead">All 100 tinyMMLU questions for both models. Three stacked
panels share one token axis: the top panel is
the reference outcome curve o_t built from all 200 recorded samples per
position; the middle panel is the smoothed reconstruction from a reduced run (only
the first S samples at the selected positions); the bottom panel is the same
reduced run raw, before smoothing. The dropdowns pick the model, the
question, the sample count S, and the observation spacing. The cost chip
shows the selected cell's generated-token budget relative to a baseline of
S=30 samples at every token (Llama) or S=30 at every sentence (DeepSeek).
Dotted vertical lines mark either the reference forks (fixed per question)
or the smoothed model's PELT change points, which depend on the selected
budget (S and spacing) — switch with the radio control.
This file is fully self-contained and can be opened locally or shared.</p>

<div class="controls">
  <div class="ctrl"><label>model / dataset</label>
    <select id="mSel"></select></div>
  <div class="ctrl"><label>question</label><select id="qSel"></select></div>
  <div class="ctrl"><label>samples per position S</label>
    <select id="sSel"></select></div>
  <div class="ctrl"><label id="spLabel">observation spacing</label>
    <select id="spSel"></select></div>
  <div class="ctrl"><label>dotted vertical lines</label>
    <span class="radios">
      <label><input type="radio" name="vlines" id="vRef" checked>
        Reference forks (TVD &gt; 0.15)</label>
      <label><input type="radio" name="vlines" id="vPelt">
        PELT change points</label>
    </span></div>
</div>
<div class="chips" id="chips"></div>
<div id="fig"></div>
<div class="tokhead">Base path — the model's greedy response for this
question, one hoverable span per token. Hover shows the 0-based response-token
index t, the same t as the panels' x-axis, so a fork at position t on the
plots maps to the token under the cursor.</div>
<div class="tokpanel" id="tokens"></div>
<div id="tokTip"></div>
"""

APP_JS = """
const TRACKS = __TRACKS__;
const CAT_COLORS = ["#C4650D", "#4E728A", "#2E6E4E", "#988453", "#B9605B"];
const CONFIG = {responsive: true, displayModeBar: "hover",
                displaylogo: false};
const mSel = document.getElementById("mSel"),
      qSel = document.getElementById("qSel"),
      sSel = document.getElementById("sSel"),
      spSel = document.getElementById("spSel"),
      spLabel = document.getElementById("spLabel"),
      chips = document.getElementById("chips"),
      tokPanel = document.getElementById("tokens"),
      tokTip = document.getElementById("tokTip"),
      vRef = document.getElementById("vRef"),
      vPelt = document.getElementById("vPelt");
const DATA = {};
Object.keys(TRACKS).forEach(key => {
  DATA[key] = JSON.parse(
      document.getElementById("data-" + key).textContent);
  const o = document.createElement("option");
  o.value = key; o.textContent = TRACKS[key];
  mSel.appendChild(o);
});

function trackData() { return DATA[mSel.value]; }

function spacingText(P, st) {
  if (P.spacing_unit === "sentences")
    return st === 1 ? "every sentence" : `every ${st} sentences`;
  const tok = st * (P.grid_tokens_per_stride || 1);
  return tok === 1 ? "every token" : `every ${tok} tokens`;
}

function rebuildQOptions() {
  const P = trackData();
  qSel.innerHTML = "";
  Object.keys(P.questions).sort((a, b) => +a - +b).forEach(r => {
    const q = P.questions[r];
    const o = document.createElement("option");
    o.value = r;
    const txt = (q.question_text || "").slice(0, 60);
    o.textContent = `row ${r} — ${txt}${(q.question_text || "").length > 60 ? "…" : ""}`;
    qSel.appendChild(o);
  });
  spLabel.textContent = "observation spacing (" +
    (P.spacing_unit === "sentences" ? "sentences" : "tokens") + ")";
  const prevSp = spSel.value;
  spSel.innerHTML = "";
  P.stride_grid.forEach(st => {
    const o = document.createElement("option");
    o.value = st;
    o.textContent = P.spacing_unit === "sentences"
        ? st : st * (P.grid_tokens_per_stride || 1);
    spSel.appendChild(o);
  });
  if ([...spSel.options].some(o => o.value === prevSp)) spSel.value = prevSp;
  else spSel.value = String(P.status_quo_stride || P.stride_grid[0]);
  rebuildSOptions();
}

function rebuildSOptions() {
  const P = trackData();
  const q = P.questions[qSel.value];
  const prev = sSel.value;
  sSel.innerHTML = "";
  P.s_grid.filter(S => S < q.n_total).forEach(S => {
    const o = document.createElement("option");
    o.value = S; o.textContent = S;
    sSel.appendChild(o);
  });
  if ([...sSel.options].some(o => o.value === prev)) sSel.value = prev;
  else sSel.value = "30";
}

function stack(x, curve, cats, axis, showLegend) {
  return cats.map((c, k) => ({
    x: x, y: curve.map(row => row[k]), name: c, legendgroup: c,
    stackgroup: "g" + axis, xaxis: "x", yaxis: axis,
    mode: "lines", line: {width: 0.6, color: CAT_COLORS[k]},
    fillcolor: CAT_COLORS[k], showlegend: showLegend,
    hovertemplate: c + " %{y:.3f} at t=%{x}<extra></extra>"
  }));
}

function render() {
  const P = trackData();
  const r = qSel.value, S = sSel.value, st = spSel.value;
  const q = P.questions[r];
  const combo = P.combos[`${r}|${S}|${st}`];
  if (!q || !combo) {
    chips.innerHTML = "<span class='chip'>combination excluded</span>";
    Plotly.purge("fig");
    return;
  }
  const p = combo.params;
  chips.innerHTML =
    `<span class="chip">Correct answer: <b>${q.answer}</b></span>` +
    `<span class="chip">cost vs status quo <b>×${combo.cost_ratio.toFixed(3)}</b></span>` +
    `<span class="chip">${combo.obs_tok.length} observed positions</span>` +
    `<span class="chip">smoothing settings (cross-validated): ${p.variant} cost, pen ${p.pen}, h ${p.h}</span>` +
    `<span class="chip">${q.ref_label} (full pool)</span>` +
    (q.flagged ? `<span class="chip">⚠ ${q.flagged}</span>` : "");
  const cats = q.categories;
  const traces = [
    ...stack(q.idxs, q.ref, cats, "y", true),
    ...stack(q.idxs, combo.pred, cats, "y2", false),
    ...stack(combo.obs_tok, combo.raw, cats, "y3", false)
  ];
  const ann = (y, text) => ({x: 0.5, y: y, xref: "paper", yref: "paper",
    xanchor: "center", yanchor: "bottom", showarrow: false,
    font: {size: 11, color: "#1D272A"}, text: text});
  const vpos = vPelt.checked ? (combo.cps || []) : q.jumps;
  const shapes = vpos.flatMap(jt => ["y", "y2", "y3"].map(ax => ({
    type: "line", x0: jt, x1: jt, xref: "x", yref: ax, y0: 0, y1: 1,
    line: {color: "#B9605B", width: 1, dash: "dot"}})));
  const axStyle = {gridcolor: "#e4ddd2", zerolinecolor: "#e4ddd2",
                   range: [0, 1], ticks: "outside",
                   title: {text: "o_t", font: {size: 12}}};
  Plotly.react("fig", traces, {
    height: 720, paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {color: "#1D272A", size: 13,
           family: "'Suisse Intl',-apple-system,Arial,sans-serif"},
    margin: {t: 34, r: 24, b: 46, l: 52},
    legend: {orientation: "h", x: 0, y: -0.09, yanchor: "top"},
    xaxis: {domain: [0, 1], anchor: "y3", gridcolor: "#e4ddd2",
            title: {text: "response token position t", font: {size: 12}}},
    yaxis:  {...axStyle, domain: [0.74, 1.0]},
    yaxis2: {...axStyle, domain: [0.37, 0.63]},
    yaxis3: {...axStyle, domain: [0.0, 0.26]},
    annotations: [
      ann(1.0,  `${q.ref_label} — full recorded pool`),
      ann(0.63, `smoothed reduced run — first ${S} draws, ` +
                `${spacingText(P, +st)} (×${combo.cost_ratio.toFixed(3)} budget)`),
      ann(0.26, `raw reduced run — same ${S} draws, before smoothing`)
    ],
    shapes: shapes
  }, CONFIG);
  renderTokens();
}

// --- tokenized base-path panel -------------------------------------------
// Custom cursor-tracking pill tooltip (no native tooltip mechanisms).
let tokKey = null;
function renderTokens() {
  const key = mSel.value + "|" + qSel.value;
  if (key === tokKey) return;
  tokKey = key;
  const q = trackData().questions[qSel.value];
  tokPanel.textContent = "";
  const frag = document.createDocumentFragment();
  (q.tokens || []).forEach((t, i) => {
    const s = document.createElement("span");
    s.className = "tok";
    s.dataset.t = i;
    s.textContent = t;
    frag.appendChild(s);
  });
  tokPanel.appendChild(frag);
}
tokPanel.addEventListener("mousemove", e => {
  tokTip.style.left = Math.min(e.clientX + 14,
                               window.innerWidth - 540) + "px";
  tokTip.style.top = (e.clientY + 18) + "px";
});
tokPanel.addEventListener("mouseover", e => {
  const t = e.target && e.target.dataset ? e.target.dataset.t : undefined;
  if (t === undefined) { tokTip.style.display = "none"; return; }
  tokTip.textContent =
      "t=" + t + ": '" + JSON.stringify(e.target.textContent).slice(1, -1)
      + "'";
  tokTip.style.display = "block";
});
tokPanel.addEventListener("mouseout", () => {
  tokTip.style.display = "none";
});

mSel.addEventListener("change", () => { rebuildQOptions(); render(); });
qSel.addEventListener("change", () => { rebuildSOptions(); render(); });
sSel.addEventListener("change", render);
spSel.addEventListener("change", render);
vRef.addEventListener("change", render);
vPelt.addEventListener("change", render);
rebuildQOptions();
if ([...qSel.options].some(o => String(o.value) === "39")) {
  qSel.value = "39";
  rebuildSOptions();
}
render();
"""


GRID_KEYS = ["s_grid", "stride_grid", "spacing_unit",
             "grid_tokens_per_stride", "status_quo_stride"]


def merge_payloads(paths):
    """Merge one or more dashboard payload files (comma-separated) for a
    track. Grids must agree; question/combo key sets must not overlap."""
    merged = None
    for path in paths.split(","):
        with open(path) as f:
            p = json.load(f)
        if merged is None:
            merged = p
            continue
        for k in GRID_KEYS:
            assert merged[k] == p[k], (k, merged[k], p[k], path)
        dup_q = set(merged["questions"]) & set(p["questions"])
        dup_c = set(merged["combos"]) & set(p["combos"])
        assert not dup_q and not dup_c, (sorted(dup_q)[:3], sorted(dup_c)[:3])
        merged["questions"].update(p["questions"])
        merged["combos"].update(p["combos"])
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama", required=True,
                    help="payload file(s), comma-separated")
    ap.add_argument("--deepseek", required=True)
    ap.add_argument("--flagged", default="")
    ap.add_argument("--tokens", required=True,
                    help="tokens_meta.json: {track: {row: {answer, tokens}}}")
    ap.add_argument("--changepoints", required=True,
                    help="cps_meta.json: {track: {combo_key: {cps, sha}}}")
    ap.add_argument("--out", default="dashboard_standalone.html")
    args = ap.parse_args()
    with open(args.tokens) as f:
        tokens_meta = json.load(f)
    with open(args.changepoints) as f:
        cps_meta = json.load(f)
    plotly_js = open(os.path.join(os.path.dirname(plotly.__file__),
                                  "package_data", "plotly.min.js")).read()
    payloads = {}
    for track, paths in (("llama", args.llama), ("deepseek", args.deepseek)):
        payloads[track] = merge_payloads(paths)
        # auto-flag near-zero answered-rate questions (payloads that carry
        # answered_rate); pre-flagged questions keep their existing note
        for row, qm in payloads[track]["questions"].items():
            ar = qm.get("answered_rate")
            if ar is not None and ar < 0.01 and not qm.get("flagged"):
                qm["flagged"] = (f"answered-rate {ar:.3f} — model almost "
                                 "never emits a committed answer; kept "
                                 "but flagged")
    for spec in [s for s in args.flagged.split(";") if s]:
        track, row, note = spec.split(":", 2)
        payloads[track]["questions"][row]["flagged"] = note

    # inject base-path tokens + ground-truth answers; assert alignment of
    # the token index space with the plots' x-axis per question
    for track, payload in payloads.items():
        for row, qm in payload["questions"].items():
            tm = tokens_meta[track][row]
            toks, ans = tm["tokens"], tm["answer"]
            assert ans in "ABCD", (track, row, ans)
            assert toks and all(isinstance(t, str) for t in toks)
            assert not any(m in t for t in toks
                           for m in ("\u0120", "\u010a", "\u010b")), \
                (track, row, "BPE marker leaked into token text")
            idxs = qm["idxs"]
            if payload["spacing_unit"] == "tokens":
                assert idxs == list(range(len(toks))), \
                    (track, row, "token panel misaligned with x-axis")
            else:
                assert all(a < b for a, b in zip(idxs, idxs[1:])) \
                    and 0 <= idxs[0] and idxs[-1] < len(toks), \
                    (track, row, "grid positions outside token range")
            qm["tokens"] = toks
            qm["answer"] = ans

    # inject per-combination PELT changepoints; the sha assertion ties the
    # served payload's smoothed curve to the exact re-fit that produced
    # these changepoints (extract_changepoints.py asserted refit == stored)
    import hashlib
    for track, payload in payloads.items():
        meta = cps_meta[track]
        assert set(meta) == set(payload["combos"]), track
        for key, combo in payload["combos"].items():
            h = hashlib.sha1(json.dumps(
                combo["pred"], separators=(",", ":")).encode()).hexdigest()
            assert h == meta[key]["sha"], (track, key,
                                           "served pred != validated pred")
            cps = meta[key]["cps"]
            assert isinstance(cps, list)
            combo["cps"] = cps

    def json_block(track):
        # </script> can't appear inside; escape the sequence defensively
        blob = json.dumps(payloads[track], separators=(",", ":"))
        blob = blob.replace("</", "<\\/")
        return (f'<script type="application/json" id="data-{track}">'
                f"{blob}</script>")

    tracks_labels = json.dumps(TRACK_META)
    html = (HTML_TOP
            + json_block("llama") + "\n" + json_block("deepseek") + "\n"
            + "<script>" + plotly_js + "</script>\n"
            + "<script>" + APP_JS.replace("__TRACKS__", tracks_labels)
            + "</script>\n</body>\n</html>\n")
    out = args.out
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
