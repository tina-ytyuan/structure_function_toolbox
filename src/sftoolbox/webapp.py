"""Local web app for the structure-function toolbox.

Runs a small Flask server on your machine. Open the printed localhost URL to:
  - upload a subject's cleaned BOLD (the only datasource for now),
  - pick an fMRI measure (ReHo, ALFF/fALFF, seed-based FC),
  - set that measure's parameters (revealed after you choose it),
  - generate the measure map and its value distribution.

Start it with:
    python -m sftoolbox.webapp
    # then open http://127.0.0.1:5000

Normal local server (nothing is uploaded anywhere); figures are rendered
in-process and embedded in the results page.
"""

from __future__ import annotations

import base64
import html
import io as _stdio
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlencode

# Headless plotting.
import matplotlib
import numpy as np

matplotlib.use("Agg")

from flask import Flask, abort, render_template_string, request, send_file

from . import fa as fa_mod
from . import io, measure_norm, measures, viz

app = Flask(__name__)

# Persistent per-session cache so a single-subject upload can be reused when
# the user switches measures on the results page (no re-upload needed).
CACHE_DIR = Path(tempfile.mkdtemp(prefix="sft_cache_"))

# ---- HTML (kept inline so the app is one file) ------------------------

STYLE = """
<style>
 :root{
   --bg:#f4f4f2; --surface:#ffffff; --border:#dcdcd8; --border-strong:#bcbcb6;
   --text:#1c1c1a; --muted:#68675f; --heading:#111110;
   --accent:#2b2a27; --accent-hover:#484640; --accent-soft:#e7e7e2;
   --secondary:#ededea; --secondary-hover:#e2e2dd; --secondary-text:#2b2a27;
   --err:#7a2c20; --err-soft:#f2e7e4;
   --radius:4px;
 }
 *{box-sizing:border-box;}
 body{font-family:Helvetica,Arial,"Helvetica Neue",sans-serif;
      margin:0;background:var(--bg);color:var(--text);line-height:1.5;
      font-size:15.5px;-webkit-font-smoothing:antialiased;}
 header{background:var(--surface);border-bottom:1px solid var(--border-strong);
        padding:1rem 1.5rem;position:sticky;top:0;z-index:5;}
 header .wrap{max-width:840px;margin:0 auto;display:flex;align-items:baseline;
              gap:.75rem;}
 header .title{font-size:1.25rem;font-weight:700;color:var(--heading);}
 header .tag{font-size:.9rem;color:var(--muted);font-style:italic;}
 main{max-width:840px;margin:0 auto;padding:1.75rem 1.5rem 3rem;}
 main.wide{max-width:1180px;}
 .lead{color:var(--muted);font-size:1rem;margin:.25rem 0 1.5rem;max-width:65ch;}
 h2{font-size:1.15rem;font-weight:700;color:var(--heading);margin:0 0 .3rem;}
 .card{background:var(--surface);border:1px solid var(--border);
       border-radius:var(--radius);padding:1.3rem 1.5rem;margin:1.1rem 0;}
 .card .sub{color:var(--muted);font-size:.92rem;margin:.1rem 0 1rem;}
 h3.sh{font-size:.82rem;font-weight:700;text-transform:uppercase;
       letter-spacing:.06em;color:var(--muted);margin:1.6rem 0 .5rem;
       padding-top:.9rem;border-top:1px solid var(--border);}
 h3.sh:first-of-type{border-top:none;padding-top:0;margin-top:1.1rem;}
 label{display:block;margin:.9rem 0 .35rem;font-weight:700;font-size:.92rem;
        color:var(--heading);}
 input[type=text],input[type=number],input[type=file],select{width:100%;
        font-family:inherit;padding:.55rem .7rem;border:1px solid var(--border-strong);
        border-radius:var(--radius);background:#fff;font-size:.95rem;color:var(--text);
        transition:border-color .15s ease,box-shadow .15s ease;}
 select{appearance:none;-webkit-appearance:none;cursor:pointer;
        background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath fill='%232b2a27' d='M1 1l5 5 5-5'/%3E%3C/svg%3E");
        background-repeat:no-repeat;background-position:right .8rem center;
        padding-right:2rem;}
 input:focus,select:focus{outline:none;border-color:var(--accent);
        box-shadow:0 0 0 2px var(--accent-soft);}
 input[type=file]{padding:.4rem .5rem;background:#fbfbfa;cursor:pointer;}
 /* Minimal slider: thin rule + small square charcoal handle. */
 input[type=range]{-webkit-appearance:none;appearance:none;width:100%;
        height:2px;background:var(--border-strong);border:none;padding:0;
        margin:.9rem 0;cursor:pointer;}
 input[type=range]:focus{outline:none;}
 input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;
        width:13px;height:13px;border-radius:2px;background:var(--accent);
        border:1px solid var(--accent);cursor:pointer;}
 input[type=range]::-moz-range-thumb{width:13px;height:13px;border-radius:2px;
        background:var(--accent);border:1px solid var(--accent);cursor:pointer;}
 input[type=range]::-moz-range-track{height:2px;background:var(--border-strong);}
 .row{display:flex;gap:1rem;flex-wrap:wrap;} .row>div{flex:1;min-width:150px;}
 .params{border-top:1px solid var(--border);margin-top:1.2rem;padding-top:.4rem;}
 .params .phint{color:var(--muted);font-size:.85rem;margin-top:.6rem;}
 .cap{color:var(--muted);font-size:.85rem;margin:.15rem 0 1.4rem 0;
   padding-left:.8rem;border-left:3px solid var(--line);max-width:60em;}
 .phint{color:var(--muted);font-size:.85rem;margin-top:.6rem;}
 table.peaks{border-collapse:collapse;margin-top:.8rem;font-size:.9rem;}
 table.peaks th,table.peaks td{border:1px solid var(--line);padding:.35rem .7rem;
   text-align:left;}
 table.peaks th{background:#f2f2ef;font-weight:600;}
 table.peaks td{font-variant-numeric:tabular-nums;}
 .mcheck{display:flex;gap:1.75rem;flex-wrap:wrap;margin:.4rem 0 .2rem;}
 .mcol{display:flex;flex-direction:column;gap:.15rem;min-width:200px;}
 .mhdr{font-size:.78rem;font-weight:700;text-transform:uppercase;
       letter-spacing:.05em;color:var(--muted);margin:.2rem 0 .3rem;}
 .mcol label{display:flex;align-items:center;gap:.5rem;margin:0;font-weight:400;
       font-size:.92rem;color:var(--text);}
 .mcol label input{width:auto;}
 .pgroup{display:none;} .pgroup.active{display:block;}
 .mgroup{display:none;} .mgroup.active{display:block;}
 .pickrow{display:flex;gap:.6rem;align-items:stretch;}
 .pickrow input{flex:1;} .pickrow button{white-space:nowrap;}
 .demorow{display:flex;gap:.7rem;flex-wrap:wrap;margin-top:1rem;}
 .actions{margin-top:1.5rem;display:flex;gap:.7rem;flex-wrap:wrap;}
 button{font-family:inherit;font-size:1rem;font-weight:700;cursor:pointer;
        border-radius:var(--radius);padding:.55rem 1.25rem;
        border:1px solid var(--accent);transition:background .15s ease;}
 .btn-primary{background:var(--accent);color:#fff;}
 .btn-primary:hover{background:var(--accent-hover);border-color:var(--accent-hover);}
 .btn-secondary{background:var(--secondary);color:var(--secondary-text);
        border-color:var(--border-strong);}
 .btn-secondary:hover{background:var(--secondary-hover);}
 a.btn-secondary,a.btn-primary{display:inline-block;text-decoration:none;
        font-weight:700;font-size:.95rem;border-radius:var(--radius);
        padding:.5rem 1.1rem;border:1px solid var(--accent);
        transition:background .15s ease;}
 a.btn-secondary{background:var(--secondary);color:var(--secondary-text);
        border-color:var(--border-strong);}
 a.btn-secondary:hover{background:var(--secondary-hover);text-decoration:none;}
 a.btn-secondary.preparing{opacity:.65;pointer-events:none;position:relative;
        padding-left:2rem;}
 a.btn-secondary.preparing::before{content:"";position:absolute;left:.8rem;
        top:50%;margin-top:-6px;width:11px;height:11px;border-radius:50%;
        border:2px solid var(--border-strong);border-top-color:var(--accent);
        animation:spin .8s linear infinite;}
 .dllink{font-size:.9rem;}
 .muted{color:var(--muted);font-size:.92rem;}
 .hint{color:var(--muted);font-size:.88rem;margin-top:.9rem;line-height:1.5;}
 img{max-width:100%;border-radius:var(--radius);display:block;}
 a{color:var(--accent);text-decoration:underline;font-weight:400;}
 a:hover{color:var(--accent-hover);}
 .back{display:inline-block;margin-bottom:.5rem;font-size:.95rem;}
 .stats{display:flex;gap:1.75rem;flex-wrap:wrap;margin-top:.4rem;}
 .stat .k{font-size:.8rem;color:var(--muted);text-transform:uppercase;
          letter-spacing:.05em;}
 .stat .v{font-size:1.35rem;font-weight:700;color:var(--heading);
          font-variant-numeric:tabular-nums;}
 .err{background:var(--err-soft);border-color:#dcbfb8;color:var(--err);}
 .err b{color:var(--err);}
 #busy{display:none;position:fixed;inset:0;z-index:100;
       background:rgba(28,28,26,.55);align-items:center;justify-content:center;}
 #busy .box{background:#fff;border:1px solid var(--border-strong);
       border-radius:var(--radius);padding:1.4rem 1.8rem;max-width:26rem;
       text-align:center;}
 #busy .box .t{font-weight:700;margin-bottom:.4rem;}
 @keyframes spin{to{transform:rotate(360deg);}}
 #busy .spin{width:22px;height:22px;margin:0 auto .7rem;border:3px solid var(--border-strong);
       border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite;}
</style>
"""

# Full-screen "Computing…" overlay shown when a compute form is submitted, so a
# slow job (MSE/ReHo/Coherence-ReHo on a large volume, or an FA re-threshold on a
# full-resolution map) doesn't look frozen.
OVERLAY = """
<div id="busy"><div class="box">
  <div class="spin"></div>
  <div class="t" id="busy-title">Computing…</div>
  <div class="muted" id="busy-note">This can take a few minutes for MSE, ReHo, or
  Coherence-ReHo on full-resolution data. Keep this tab open.</div>
</div></div>
<script>
(function(){
  function showBusy(title, note){
    var b = document.getElementById('busy');
    if (!b) return;
    if (title) document.getElementById('busy-title').textContent = title;
    if (note) document.getElementById('busy-note').textContent = note;
    b.style.display = 'flex';
  }
  document.querySelectorAll('form').forEach(function(f){
    var a = f.getAttribute('action') || '';
    if (a.indexOf('/run') !== -1 || a.indexOf('/demo') !== -1) {
      f.addEventListener('submit', function(){ showBusy(); });
    } else if (a.indexOf('/fa') !== -1) {
      f.addEventListener('submit', function(){
        showBusy('Processing FA map…',
          'Applying the threshold and rendering slices. Regional FA over a ' +
          'full atlas can take a moment. Keep this tab open.');
      });
    }
  });

  // Server-side downloads (NIfTI/zip) are recomputed on request, so show an
  // inline "preparing" state on the clicked button until the file arrives.
  // A cookie set by the server tells us the download actually started.
  document.querySelectorAll('a[data-prepare]').forEach(function(a){
    a.addEventListener('click', function(){
      if (a.dataset.busy === '1') return;
      a.dataset.busy = '1';
      var original = a.textContent;
      a.classList.add('preparing');
      a.textContent = 'Preparing…';
      var done = false;
      function finish(){
        if (done) return;
        done = true;
        a.classList.remove('preparing');
        a.textContent = original;
        a.dataset.busy = '';
      }
      // Clear when the file lands (cookie flips) or after a generous timeout.
      var token = 'dl_' + Math.random().toString(36).slice(2);
      var started = Date.now();
      var poll = setInterval(function(){
        if (document.cookie.indexOf('sft_dl=') !== -1) {
          document.cookie = 'sft_dl=; Max-Age=0; path=/';
          clearInterval(poll); finish();
        } else if (Date.now() - started > 600000) {
          clearInterval(poll); finish();
        }
      }, 500);
      void token;
    });
  });
})();
</script>
"""

PAGE = (
    """
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Structure-Function Toolbox</title>
"""
    + STYLE
    + """
</head><body>
<header><div class="wrap">
  <span class="title">Structure-Function Toolbox</span>
  <span class="tag">local &middot; fMRI measures</span>
</div></header>
<main>
<p class="lead">Upload a subject's cleaned BOLD, check one or more fMRI measures,
set their parameters, and generate all of them at once. Everything runs on your
machine.</p>

{% if error %}<div class="card err"><b>Problem:</b> {{ error }}</div>{% endif %}

<form method="post" action="/run" enctype="multipart/form-data" class="card">
  <h2>1 &nbsp;Data</h2>
  <p class="sub">Cleaned, preprocessed resting-state BOLD is the only input.</p>

  <label>Input mode</label>
  <select name="mode" id="mode" onchange="showMode()">
    <option value="folder" selected>Folder of subjects &mdash; choose a folder</option>
    <option value="single">Single subject &mdash; upload a BOLD file</option>
  </select>

  <div class="mgroup" data-mode="single" id="mode-single">
    <label>Subject ID</label>
    <input type="text" name="subject_id" value="subject-01">
    <label>Cleaned BOLD &mdash; 4D NIfTI</label>
    <input type="file" name="bold">
    <label>Mask &mdash; 3D NIfTI (optional, recommended)</label>
    <input type="file" name="mask">
    <p class="phint">Restricts the analysis to voxels &ne; 0 in this mask,
    instead of the default finite/nonzero mask. <b>Use the same mask the
    reference cohort was built with</b> so the comparison is like-for-like; the
    results page reports the cohort's mask and flags a mismatch. It changes the
    reported summary statistics for every measure, and changes the values
    themselves for neighbourhood measures (ReHo, Coherence-ReHo), whose voxels
    depend on which neighbours are included.</p>
  </div>

  <div class="mgroup" data-mode="folder" id="mode-folder">
    <label>Subjects folder</label>
    <div class="pickrow">
      <input type="text" name="folder" id="folder" placeholder="Choose a folder…">
      <button type="button" class="btn-secondary" onclick="pickFolder()">Choose folder…</button>
    </div>
    <p class="phint">Expects one subfolder per subject (each containing a BOLD
    NIfTI), or a flat folder of BOLD files. Read directly from disk &mdash;
    nothing is uploaded. Processes up to {{ max_batch }} subjects.</p>
  </div>

  <div class="params">
    <h2 style="margin-top:.6rem;">2 &nbsp;fMRI measures</h2>
    <p class="sub">Check one or more. Checked measures reveal their parameters
    below, and all are generated together.</p>
    <div class="mcheck">
      <div class="mcol">
        <div class="mhdr">Frequency-based</div>
        <label><input type="checkbox" name="measures" value="alff" onchange="showParams()"> ALFF (broadband)</label>
        <label><input type="checkbox" name="measures" value="falff" onchange="showParams()"> fALFF (broadband)</label>
        <label><input type="checkbox" name="measures" value="alff_slow5" onchange="showParams()"> ALFF (slow-5)</label>
        <label><input type="checkbox" name="measures" value="alff_slow4" onchange="showParams()"> ALFF (slow-4)</label>
        <label><input type="checkbox" name="measures" value="falff_slow5" onchange="showParams()"> fALFF (slow-5)</label>
        <label><input type="checkbox" name="measures" value="falff_slow4" onchange="showParams()"> fALFF (slow-4)</label>
      </div>
      <div class="mcol">
        <div class="mhdr">Local synchrony</div>
        <label><input type="checkbox" name="measures" value="reho" onchange="showParams()"> Regional Homogeneity (ReHo)</label>
        <label><input type="checkbox" name="measures" value="coherence_reho" onchange="showParams()"> Coherence Regional Homogeneity</label>
        <label><input type="checkbox" name="measures" value="rsfa" onchange="showParams()"> RSFA</label>
        <label><input type="checkbox" name="measures" value="int" onchange="showParams()"> Intrinsic Neural Timescale (INT)</label>
      </div>
      <div class="mcol">
        <div class="mhdr">Entropy</div>
        <label><input type="checkbox" name="measures" value="mse" onchange="showParams()"> Multiscale Entropy (MSE)</label>
      </div>
    </div>
    <p class="phint">MSE, ReHo, and Coherence-ReHo are computed per voxel and can
    take several minutes on a full-resolution brain; upload a mask below to
    restrict them, or run heavy measures on a cluster. The frequency measures are
    fast.</p>

    <!-- Shared acquisition params (shown only when a measure is checked) -->
    <div class="needs-measure" style="display:none;">
    <div class="row">
      <div><label>TR (seconds)</label>
           <input type="number" name="alff_tr" step="0.01" value="0.72"></div>
      <div><label>Low (Hz)</label>
           <input type="number" name="alff_low" step="0.001" value="0.01"></div>
      <div><label>High (Hz)</label>
           <input type="number" name="alff_high" step="0.001" value="0.08"></div>
    </div>
    <p class="phint">TR sets the frequency axis. Low/High define the band for
    ALFF, fALFF, RSFA, and Coherence-ReHo; the slow-4/slow-5 measures use their
    own fixed bands.</p>
    </div>

    <!-- ReHo params -->
    <div class="pgroup" data-measure="reho">
      <label>Cluster size</label>
      <select name="reho_cluster">
        <option value="7">7 voxels (faces)</option>
        <option value="19">19 voxels (+ edges)</option>
        <option value="27" selected>27 voxels (+ corners)</option>
      </select>
      <p class="phint">Neighborhood over which local time-series concordance
      (Kendall's W) is computed.</p>
    </div>

    <!-- INT params -->
    <div class="pgroup" data-measure="int">
      <label>Max lag (TRs)</label>
      <input type="number" name="int_max_lag" step="1" value="20">
      <p class="phint">Autocorrelation lags summed until the first non-positive
      value (Watanabe et al. 2019).</p>
    </div>

    <!-- MSE params -->
    <div class="pgroup" data-measure="mse">
      <label>Scales (space-separated)</label>
      <input type="text" name="mse_scales" value="1 2 3 4 5">
      <div class="row">
        <div><label>m (embedding)</label>
             <input type="number" name="mse_m" step="1" value="2"></div>
        <div><label>r (tolerance ratio)</label>
             <input type="number" name="mse_r" step="0.01" value="0.15"></div>
      </div>
      <p class="phint">Sample entropy per coarse-grained scale; the map is the
      NaN-aware mean across scales (complexity index). Requires antropy.</p>
    </div>
  </div>

    <div class="needs-measure" style="display:none;">
    <label style="font-weight:400;display:flex;align-items:center;gap:.5rem;margin-top:1rem;">
      <input type="checkbox" name="include_zeros" value="on" style="width:auto;">
      Include zero-valued voxels in the value distribution
    </label>
    <p class="phint">Off by default: exact-zero voxels (mask voxels the measure
    couldn't compute, or image edges) otherwise pile up as a spike at 0.</p>
    </div>
  </div>

  <div class="actions">
    <button type="submit" class="btn-primary">Generate measures</button>
  </div>
</form>

<!-- The FA / structural panel is not shown. The /fa route and sftoolbox.fa
     remain available, so re-enabling it is a matter of restoring this form. -->

<form method="post" action="/demo" class="card">
  <h2>No data handy?</h2>
  <p class="sub">Run any measure on synthetic BOLD to preview its output.</p>
  <div class="demorow">
    <button type="submit" name="measure" value="reho" class="btn-secondary">Demo: ReHo</button>
    <button type="submit" name="measure" value="alff" class="btn-secondary">Demo: ALFF</button>
    <button type="submit" name="measure" value="falff" class="btn-secondary">Demo: fALFF</button>
    <button type="submit" name="measure" value="rsfa" class="btn-secondary">Demo: RSFA</button>
    <button type="submit" name="measure" value="int" class="btn-secondary">Demo: INT</button>
    <button type="submit" name="measure" value="coherence_reho" class="btn-secondary">Demo: Coherence-ReHo</button>
    <button type="submit" name="measure" value="mse" class="btn-secondary">Demo: MSE</button>
  </div>
  <p class="phint">MSE requires the antropy package and is the slowest to
  compute; Coherence-ReHo is also slower than the frequency measures.</p>
</form>

<script>
function pickFolder(){
  fetch('/pick-folder').then(function(r){return r.json();}).then(function(d){
    if(d.path){ document.getElementById('folder').value = d.path; }
    else if(d.error){ alert('Folder picker unavailable: ' + d.error +
      '\\nYou can paste the path manually.'); }
  }).catch(function(e){ alert('Could not open picker: ' + e); });
}
function checkedMeasures(){
  return Array.prototype.map.call(
    document.querySelectorAll('input[name="measures"]:checked'),
    function(c){ return c.value; });
}
function showParams(){
  var checked = checkedMeasures();
  var any = checked.length > 0;
  document.querySelectorAll('.needs-measure').forEach(function(el){
    el.style.display = any ? '' : 'none';
  });
  document.querySelectorAll('.pgroup').forEach(function(g){
    var ms = g.getAttribute('data-measure').split(' ');
    var on = ms.some(function(x){ return checked.indexOf(x) !== -1; });
    g.classList.toggle('active', on);
  });
}
function showMode(){
  var m = document.getElementById('mode').value;
  document.querySelectorAll('.mgroup').forEach(function(g){
    g.classList.toggle('active', g.getAttribute('data-mode') === m);
  });
}
showParams(); showMode();
</script>
</main>
"""
    + OVERLAY
    + """
</body></html>
"""
)

RESULT = (
    """
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Results: {{ sid }}</title>
"""
    + STYLE
    + """
</head><body>
<header><div class="wrap">
  <span class="title">Structure-Function Toolbox</span>
  <span class="tag">results</span>
</div></header>
<main>
<a class="back" href="/">&larr; New analysis</a>
<h2 style="font-size:1.25rem;margin:.3rem 0 .2rem;">{{ sid }}</h2>
<p class="muted">{{ blocks|length }} measure{{ '' if blocks|length == 1 else 's' }}</p>

{% for b in blocks %}
<div class="card">
  <h2>{{ b.measure_label }}</h2>
  {% if b.params %}<p class="sub">{{ b.params }}</p>{% endif %}

  <div class="actions" style="margin-top:.7rem;">
    {% if b.download_url %}
    <a class="btn-secondary" data-prepare="1" href="{{ b.download_url }}">Measure map (.nii.gz)</a>
    {% endif %}
    {% if b.stats_url %}
    <a class="btn-secondary" data-prepare="1" href="{{ b.stats_url }}">t, p &amp; q maps (.nii.gz)</a>
    {% endif %}
    {% if b.mask_url %}
    <a class="btn-secondary" data-prepare="1" href="{{ b.mask_url }}">Cohort mask (.nii.gz)</a>
    {% endif %}
    <a class="btn-secondary" download="{{ sid }}_{{ b.measure_key }}.png"
       href="data:image/png;base64,{{ b.map_png }}">Measure image (PNG)</a>
  </div>

  <h3 class="sh">Measure map</h3>
  <p class="sub">Axial, coronal, and sagittal slices through the measure map.</p>
  <img src="data:image/png;base64,{{ b.map_png }}">
  <div class="stats">
    <div class="stat"><div class="k">mean</div><div class="v">{{ b.mean }}</div></div>
    <div class="stat"><div class="k">median</div><div class="v">{{ b.median }}</div></div>
    <div class="stat"><div class="k">masked voxels</div><div class="v">{{ b.nvox }}</div></div>
  </div>

  <h3 class="sh">Value distribution</h3>
  <img src="data:image/png;base64,{{ b.hist_png }}">

  {% set cmp = b.cmp %}
  {% if cmp and cmp.mode == 't' %}
    <h3 class="sh">Comparison to group (t-test)</h3>
    <p class="sub">Single-subject vs group t-test (Crawford &amp; Howell) against
    {{ cmp.n }} reference subjects, df = {{ cmp.df }}. Positive t = above the
    group.</p>
    <div class="stats">
      <div class="stat"><div class="k">mean t</div><div class="v">{{ cmp.t_mean }}</div></div>
      <div class="stat"><div class="k">max |t|</div><div class="v">{{ cmp.t_absmax }}</div></div>
      <div class="stat"><div class="k">voxels tested</div><div class="v">{{ cmp.n_tested }}</div></div>
    </div>
    <div class="stats" style="margin-top:.9rem;">
      <div class="stat"><div class="k">p&lt;0.05 uncorrected</div><div class="v">{{ cmp.n_sig }}</div></div>
      <div class="stat"><div class="k">FDR q&lt;0.05</div><div class="v">{{ cmp.n_fdr }}</div></div>
      <div class="stat"><div class="k">% surviving FDR</div><div class="v">{{ cmp.pct_fdr }}</div></div>
      <div class="stat"><div class="k">SD of t <span title="1.0 = correctly calibrated">(exp. 1.0)</span></div><div class="v">{{ cmp.t_sd }}</div></div>
      <div class="stat"><div class="k">observed / chance <span title="1.0 = as many p&lt;0.05 voxels as chance predicts">(exp. 1.0)</span></div><div class="v">{{ cmp.obs_exp }}</div></div>
    </div>
    {% if cmp.profile_png %}
    <img src="data:image/png;base64,{{ cmp.profile_png }}" style="margin-top:1rem;">
    <p class="cap">The shaded band is the cohort mean ±1 SD of each slice's
    average. The subject leaving the band marks a height where the whole
    slice runs high or low. Slice averaging cancels most voxel-level noise,
    so this band is tighter than the voxelwise one behind the t-test.</p>
    {% endif %}
    <img src="data:image/png;base64,{{ cmp.t_png }}" style="margin-top:1rem;">
    <p class="cap">Uncorrected. Shown for reference only; report the
    FDR-corrected result below.</p>
    {% if cmp.resampled %}
    <p class="phint"><b>Resampled:</b> {{ cmp.resampled }}</p>
    {% endif %}
    {% if cmp.calib %}
    <p class="phint" style="color:var(--err);"><b>Calibration warning:</b> {{ cmp.calib }}</p>
    {% endif %}
    {% if cmp.sparse %}
    <p class="sub" style="margin-top:1rem;">FDR-corrected result (q&lt;0.05). BH
    threshold on raw p: {{ cmp.p_thr }}. Too few voxels survive to make a useful
    figure, so they are listed individually:</p>
    <table class="peaks">
      <tr><th>MNI (x, y, z)</th><th>t</th><th>q</th><th>direction</th></tr>
      {% for pk in cmp.peaks %}
      <tr><td>{{ pk.xyz }}</td><td>{{ pk.t }}</td><td>{{ pk.q }}</td>
          <td>{{ pk.dir }} the group</td></tr>
      {% endfor %}
    </table>
    <p class="phint">A result this sparse is usually indistinguishable from
    noise. Check the calibration figures above before interpreting it.</p>
    {% else %}
    <img src="data:image/png;base64,{{ cmp.t_fdr_png }}" style="margin-top:1rem;">
    <p class="cap">Non-surviving voxels set to 0. Benjamini-Hochberg
    threshold on raw p: {{ cmp.p_thr }}.</p>
    {% endif %}
    <p class="phint">Report the FDR-corrected result. Benjamini-Hochberg controls
    the expected proportion of false positives across the {{ cmp.n_tested }}
    tested voxels; the uncorrected map is shown for reference only.
    Cohort references: {% if cmp.normalized %}globally normalised (DPABI
    <i>m</i> convention){% else %}<b>raw units, not normalised</b>{% endif %}.</p>
    {% if cmp.mask_warn %}
    <p class="phint" style="color:var(--err);"><b>Mask mismatch:</b> {{ cmp.mask_warn }}</p>
    {% elif cmp.ref_mask_vox %}
    <p class="phint">Cohort mask{% if cmp.ref_mask_name %}: {{ cmp.ref_mask_name }}{% endif %}
    ({{ cmp.ref_mask_vox }} voxels), matching this subject. Download it above to
    reproduce this analysis over the same voxels.</p>
    {% endif %}
  {% elif cmp and cmp.mode == 'z' %}
    <h3 class="sh">Comparison to cohort</h3>
    {% if cmp.compare_png %}
    <div class="stats">
      <div class="stat"><div class="k">cohort percentile</div><div class="v">{{ cmp.pct }}</div></div>
    </div>
    <img src="data:image/png;base64,{{ cmp.compare_png }}" style="margin-top:.6rem;">
    {% endif %}
    {% if cmp.map_png %}
    <p class="sub" style="margin-top:1.1rem;">Deviation z-map: (subject &minus;
    cohort mean) / SD. Red = above, blue = below.</p>
    <img src="data:image/png;base64,{{ cmp.map_png }}">
    {% endif %}
  {% elif cmp and cmp.mode == 'diff' %}
    <h3 class="sh">Comparison to group average</h3>
    <div class="stats">
      <div class="stat"><div class="k">subject mean</div><div class="v">{{ cmp.subj_mean }}</div></div>
      <div class="stat"><div class="k">group mean</div><div class="v">{{ cmp.grp_mean }}</div></div>
      <div class="stat"><div class="k">difference</div><div class="v">{{ cmp.diff_mean }}</div></div>
    </div>
    {% if cmp.map_png %}
    <p class="sub" style="margin-top:1.1rem;">Difference from group average
    (subject &minus; mean). Red = above average, blue = below.</p>
    <img src="data:image/png;base64,{{ cmp.map_png }}">
    {% endif %}
    <p class="phint">No across-subject SD in this reference, so this is a raw
    difference. Rebuild the reference from individual subjects to get t/p maps.</p>
  {% endif %}
  {% if cmp and cmp.note %}<p class="muted" style="margin-top:.8rem;">{{ cmp.note }}</p>{% endif %}
</div>
{% endfor %}

{% if notes %}<div class="card muted">{{ notes }}</div>{% endif %}

{{ controls|safe }}
</main>
"""
    + OVERLAY
    + """
</body></html>
"""
)

BATCH = (
    """
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Batch results</title>
"""
    + STYLE
    + """
<style>
 table{width:100%;border-collapse:collapse;font-size:.88rem;}
 th,td{text-align:left;padding:.5rem .6rem;border-bottom:1px solid var(--border);}
 th{color:var(--muted);font-weight:600;font-size:.76rem;text-transform:uppercase;
    letter-spacing:.04em;}
 td.num{font-variant-numeric:tabular-nums;color:var(--heading);}
 td.viewcell{white-space:nowrap;text-align:right;}
 td.viewcell a{white-space:nowrap;font-size:.82rem;}
 .layout{display:flex;gap:1.5rem;align-items:flex-start;}
 .sidebar{position:sticky;top:66px;flex:0 0 175px;
          max-height:calc(100vh - 88px);overflow:auto;
          border-right:1px solid var(--border);padding-right:.4rem;}
 .sidebar h3{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;
             color:var(--muted);margin:.2rem 0 .5rem;}
 .sidebar a{display:block;padding:.35rem .5rem;border-radius:var(--radius);font-size:.88rem;
            font-weight:400;color:var(--secondary-text);text-decoration:none;
            white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
 .sidebar a:hover{background:var(--secondary);}
 .sidebar a.top{color:var(--accent);font-weight:700;margin-bottom:.3rem;}
 .maincol{flex:1;min-width:0;}
 .rightbar{position:sticky;top:66px;flex:0 0 235px;
           max-height:calc(100vh - 88px);overflow:auto;}
 .rightbar .card{margin-top:0;}
 .subject-card{scroll-margin-top:74px;}
 #backtop{position:fixed;bottom:1.5rem;right:1.5rem;z-index:30;display:none;}
 @media(max-width:1000px){.rightbar{display:none;}}
 @media(max-width:680px){.sidebar{display:none;}}
</style>
</head><body>
<header><div class="wrap">
  <span class="title">Structure-Function Toolbox</span>
  <span class="tag">batch results</span>
</div></header>
<main id="top" class="wide">
<a class="back" href="/">&larr; New analysis</a>
<h2 style="font-size:1.25rem;margin:.3rem 0 .2rem;">{{ n }} subjects</h2>
<p class="muted">{{ measure_label }}{% if params %} &middot; {{ params }}{% endif %}</p>

<div class="layout">
  <nav class="sidebar">
    <a class="top" href="#top">&uarr; Overview</a>
    {% if group_png %}<a href="#groupavg">Group average</a>{% endif %}
  </nav>

  <div class="maincol">
    <div class="card">
      <h2>Cohort</h2>
      <p class="sub">Group-level results only. Individual subject maps and
      per-subject statistics are not shown, to keep the cohort de-identified.</p>
      <div class="stats">
        <div class="stat"><div class="k">subjects in group</div>
             <div class="v">{{ n }}</div></div>
        {% if group_png %}
        <div class="stat"><div class="k">group mean</div>
             <div class="v">{{ group_mean }}</div></div>
        <div class="stat"><div class="k">group median</div>
             <div class="v">{{ group_median }}</div></div>
        {% endif %}
      </div>
    </div>

    {% if group_png %}
    <div class="card subject-card" id="groupavg">
      <h2>Group average &middot; {{ n }} subjects</h2>
      <p class="sub">Voxelwise mean of {{ measure_label }} across all subjects.</p>
      <img src="data:image/png;base64,{{ group_png }}">
      <img src="data:image/png;base64,{{ group_hist_png }}" style="margin-top:1rem;">
      <div class="actions">
        <a class="btn-secondary" href="{{ group_dl_url }}">Download group average (.nii.gz)</a>
        <a class="btn-secondary" download="group_{{ measure_key }}_average.png"
           href="data:image/png;base64,{{ group_png }}">Download image (PNG)</a>
      </div>
    </div>
    {% elif group_note %}
    <div class="card muted">{{ group_note }}</div>
    {% endif %}

    {% if notes %}<div class="card muted">{{ notes }}</div>{% endif %}
  </div>

  <aside class="rightbar">
    {{ controls|safe }}
  </aside>
</div>

<button id="backtop" class="btn-primary"
        onclick="window.scrollTo({top:0,behavior:'smooth'})">&uarr; Back to top</button>
<script>
window.addEventListener('scroll', function(){
  document.getElementById('backtop').style.display =
    window.scrollY > 300 ? 'block' : 'none';
});
</script>
</main>
</body></html>
"""
)


FA_RESULT = (
    """
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FA: {{ sid }}</title>
"""
    + STYLE
    + """
</head><body>
<header><div class="wrap">
  <span class="title">Structure-Function Toolbox</span>
  <span class="tag">FA &middot; structural</span>
</div></header>
<main>
<a class="back" href="/">&larr; New analysis</a>
<h2 style="font-size:1.25rem;margin:.3rem 0 .2rem;">{{ sid }}</h2>
<p class="muted">Fractional anisotropy &middot; threshold {{ thr }}</p>

{% if error %}<div class="card err"><b>Problem:</b> {{ error }}</div>{% endif %}

<div class="card">
  <h2>FA threshold</h2>
  <p class="sub">Voxels with FA below the threshold are excluded (treated as
  non-white-matter). Drag to re-threshold &mdash; no re-upload needed.</p>
  <form method="post" action="/fa" id="thrform">
    <input type="hidden" name="cached_fa" value="{{ cached_fa }}">
    <input type="hidden" name="cached_atlas" value="{{ cached_atlas }}">
    <input type="hidden" name="sid" value="{{ sid }}">
    <label>FA threshold: <span id="fathr_val">{{ thr }}</span></label>
    <input type="range" name="fa_threshold" id="fathr" min="0" max="0.9"
           step="0.01" value="{{ thr }}" style="width:100%;"
           oninput="document.getElementById('fathr_val').textContent =
                    parseFloat(this.value).toFixed(2);"
           onchange="var f=document.getElementById('thrform');
                     if (f.requestSubmit) { f.requestSubmit(); } else { f.submit(); }">
    <p class="phint">Release the slider to recompute at the new threshold.</p>
  </form>
</div>

<div class="card">
  <h2>FA map</h2>
  <p class="sub">Axial, coronal, and sagittal slices of the thresholded FA map.</p>
  <div class="actions" style="margin-top:.2rem;margin-bottom:1rem;">
    {% if fa_dl_url %}
    <a class="btn-secondary" data-prepare="1" href="{{ fa_dl_url }}">Thresholded FA (.nii.gz)</a>
    {% endif %}
    <a class="btn-secondary" download="{{ sid }}_FA_thr{{ thr }}.png"
       href="data:image/png;base64,{{ map_png }}">FA image (PNG)</a>
    {% if roi_png %}
    <a class="btn-secondary" download="{{ sid }}_regional_FA.png"
       href="data:image/png;base64,{{ roi_png }}">Regional FA image (PNG)</a>
    {% endif %}
  </div>
  <img src="data:image/png;base64,{{ map_png }}">
  <div class="stats">
    <div class="stat"><div class="k">mean FA (kept)</div><div class="v">{{ mean }}</div></div>
    <div class="stat"><div class="k">voxels kept</div><div class="v">{{ nvox }}</div></div>
    <div class="stat"><div class="k">% of brain kept</div><div class="v">{{ pct_kept }}</div></div>
  </div>
  <img src="data:image/png;base64,{{ hist_png }}" style="margin-top:1rem;">
</div>

{% if roi_png %}
<div class="card">
  <h2>Regional FA</h2>
  <p class="sub">Mean FA within each of {{ n_regions }} atlas parcels (voxels
  below threshold excluded from each region's mean).</p>
  <img src="data:image/png;base64,{{ roi_png }}">
</div>
{% elif roi_note %}
<div class="card muted">{{ roi_note }}</div>
{% endif %}
</main>
"""
    + OVERLAY
    + """
</body></html>
"""
)


# ---- helpers ----------------------------------------------------------

MAX_BATCH = 24


def _fig_to_b64(fig) -> str:
    buf = _stdio.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _save_upload(file_storage, dest_dir: Path):
    if file_storage is None or file_storage.filename == "":
        return None
    p = dest_dir / file_storage.filename
    file_storage.save(str(p))
    return p


def _form_params(form):
    """Collect measure parameters from a request form into a params dict."""
    def g(k, d):
        v = form.get(k) if form is not None else None
        return d if v in (None, "") else v
    scales = g("mse_scales", "1 2 3 4 5")
    try:
        mse_scales = tuple(int(s) for s in str(scales).replace(",", " ").split())
    except ValueError:
        mse_scales = measures.DEFAULT_MSE_SCALES
    return {
        "tr": float(g("alff_tr", 0.72)),
        "low": float(g("alff_low", 0.01)),
        "high": float(g("alff_high", 0.08)),
        "cluster": int(g("reho_cluster", 27)),
        "max_lag": int(g("int_max_lag", measures.DEFAULT_INT_MAX_LAG)),
        "mse_scales": mse_scales or measures.DEFAULT_MSE_SCALES,
        "mse_m": int(g("mse_m", measures.DEFAULT_MSE_M)),
        "mse_r": float(g("mse_r", measures.DEFAULT_MSE_R)),
    }


def _params_str(measure, p):
    """Human-readable parameter summary for the results header."""
    tr, lo, hi = p["tr"], p["low"], p["high"]
    if measure == "reho":
        return f"cluster {p['cluster']}"
    if measure in ("alff", "falff"):
        return f"TR {tr}s, band {lo}-{hi} Hz"
    if measure == "rsfa":
        return f"TR {tr}s, detrend + bandpass {lo}-{hi} Hz"
    if measure in ("alff_slow5", "falff_slow5"):
        return f"TR {tr}s, slow-5 0.01-0.027 Hz"
    if measure in ("alff_slow4", "falff_slow4"):
        return f"TR {tr}s, slow-4 0.027-0.073 Hz"
    if measure == "int":
        return f"TR {tr}s, max lag {p['max_lag']}"
    if measure == "coherence_reho":
        return f"TR {tr}s, band {lo}-{hi} Hz"
    if measure == "mse":
        sc = " ".join(str(s) for s in p["mse_scales"])
        return f"scales [{sc}], m={p['mse_m']}, r={p['mse_r']}"
    return ""


def _compute(measure: str, bold: np.ndarray, form, mask=None):
    """Dispatch to the chosen measure; return (map3d, mask, params_str, cmap, sym).

    ``mask`` is an optional 3D array restricting the analysis (else the default
    finite/nonzero mask is used).
    """
    if measure not in measures.MEASURES:
        raise ValueError(f"unknown measure: {measure}")
    p = _form_params(form)
    if mask is not None:
        p["mask"] = mask
    m, mask_out = measures.compute(measure, bold, p)
    cmap = viz._MEASURE_CMAP.get(measure, "magma")
    return m, mask_out, _params_str(measure, p), cmap, False


def _controls_html(mode, selected, folder=None, cached=None, form=None, sid=None,
                   cached_mask=None):
    """A compact 'change measures & regenerate' form for the results pages.

    Carries the datasource (folder path, cached upload, or folder+subject) as
    hidden fields so the user never re-selects it. In demo mode it re-runs /demo.
    """
    g = form.get if form is not None else (lambda k, d=None: d)
    reho_c = g("reho_cluster", "27")
    inc_zeros = " checked" if _truthy(g("include_zeros", "")) else ""
    tr = g("alff_tr", "0.72")
    lo = g("alff_low", "0.01")
    hi = g("alff_high", "0.08")

    action = {"demo": "/demo", "subject": "/subject"}.get(mode, "/run")
    hidden = f'<input type="hidden" name="mode" value="{mode}">'
    if mode in ("folder", "subject") and folder:
        hidden += f'<input type="hidden" name="folder" value="{html.escape(folder)}">'
    if mode == "subject" and sid:
        hidden += f'<input type="hidden" name="sid" value="{html.escape(sid)}">'
    if mode == "single" and cached:
        hidden += (
            f'<input type="hidden" name="cached_bold" value="{html.escape(cached)}">'
        )
    if mode == "single" and cached_mask:
        hidden += (
            f'<input type="hidden" name="cached_mask" value="{html.escape(cached_mask)}">'
        )
    same = {
        "demo": "same synthetic data",
        "folder": "same folder",
        "subject": "same subject",
    }.get(mode, "same subject")

    sel = set(selected if isinstance(selected, (list, tuple, set)) else [selected])

    def rsel(v):
        return " selected" if reho_c == v else ""

    def cb(v, label):
        c = " checked" if v in sel else ""
        return (f'<label><input type="checkbox" name="measures" value="{v}"'
                f'{c} onchange="showParams()"> {label}</label>')

    mse_scales = g("mse_scales", "1 2 3 4 5")
    mse_m = g("mse_m", "2")
    mse_r = g("mse_r", "0.15")
    max_lag = g("int_max_lag", "20")

    freq_cb = (cb("alff", "ALFF (broadband)") + cb("falff", "fALFF (broadband)")
               + cb("alff_slow5", "ALFF (slow-5)") + cb("alff_slow4", "ALFF (slow-4)")
               + cb("falff_slow5", "fALFF (slow-5)") + cb("falff_slow4", "fALFF (slow-4)"))
    sync_cb = (cb("reho", "Regional Homogeneity (ReHo)")
               + cb("coherence_reho", "Coherence Regional Homogeneity")
               + cb("rsfa", "RSFA") + cb("int", "Intrinsic Neural Timescale (INT)"))
    entropy_cb = cb("mse", "Multiscale Entropy (MSE)")
    checks = (f'<div class="mcol"><div class="mhdr">Frequency-based</div>{freq_cb}</div>'
              f'<div class="mcol"><div class="mhdr">Local synchrony</div>{sync_cb}</div>'
              f'<div class="mcol"><div class="mhdr">Entropy</div>{entropy_cb}</div>')

    return f'''
<form method="post" action="{action}" class="card">
  <h2>Change measures</h2>
  <p class="sub">Adjust the checked measures or their parameters and regenerate &mdash; {same}, no re-selecting.</p>
  {hidden}
  <div class="mcheck">{checks}</div>

  <div class="needs-measure" style="display:none;">
  <div class="row">
    <div><label>TR (s)</label><input type="number" name="alff_tr" step="0.01" value="{tr}"></div>
    <div><label>Low (Hz)</label><input type="number" name="alff_low" step="0.001" value="{lo}"></div>
    <div><label>High (Hz)</label><input type="number" name="alff_high" step="0.001" value="{hi}"></div>
  </div>
  </div>
  <div class="pgroup" data-measure="reho">
    <label>Cluster size</label>
    <select name="reho_cluster">
      <option value="7"{rsel("7")}>7 voxels (faces)</option>
      <option value="19"{rsel("19")}>19 voxels (+ edges)</option>
      <option value="27"{rsel("27")}>27 voxels (+ corners)</option>
    </select>
  </div>
  <div class="pgroup" data-measure="int">
    <label>Max lag (TRs)</label>
    <input type="number" name="int_max_lag" step="1" value="{max_lag}">
  </div>
  <div class="pgroup" data-measure="mse">
    <label>Scales</label>
    <input type="text" name="mse_scales" value="{mse_scales}">
    <div class="row">
      <div><label>m</label><input type="number" name="mse_m" step="1" value="{mse_m}"></div>
      <div><label>r</label><input type="number" name="mse_r" step="0.01" value="{mse_r}"></div>
    </div>
  </div>
  <div class="needs-measure" style="display:none;">
  <label style="font-weight:400;display:flex;align-items:center;gap:.5rem;margin-top:.9rem;">
    <input type="checkbox" name="include_zeros" value="on"{inc_zeros} style="width:auto;">
    Include zero-valued voxels in the value distribution
  </label>
  </div>

  <div class="actions"><button class="btn-primary">Regenerate</button></div>
  <script>
  function showParams(){{
    var checked=Array.prototype.map.call(
      document.querySelectorAll('input[name="measures"]:checked'),
      function(c){{ return c.value; }});
    var any=checked.length>0;
    document.querySelectorAll('.needs-measure').forEach(function(el){{
      el.style.display = any ? '' : 'none';
    }});
    document.querySelectorAll('.pgroup').forEach(function(g){{
      var ms=g.getAttribute('data-measure').split(' ');
      g.classList.toggle('active', ms.some(function(x){{return checked.indexOf(x)!==-1;}}));
    }});
  }}
  showParams();
  </script>
</form>'''


def _measure_label(measure):
    return measures.MEASURES.get(measure, {}).get("label", measure)


def _slices_png(map3d, affine, measure, symmetric=False, cmap=None, title=None):
    cmap = cmap or viz._MEASURE_CMAP.get(measure, "cold_hot")
    # Title every figure with the measure and what the values are, so a
    # downloaded PNG is interpretable without the surrounding page.
    if title is None:
        title = f"{_measure_label(measure)}: value per voxel"
    return _fig_to_b64(
        viz.plot_orientations(map3d, affine, cmap=cmap, symmetric=symmetric,
                              title=title)
    )


def _truthy(v):
    """Interpret an HTML checkbox / query value as a boolean."""
    return str(v).lower() in ("on", "true", "1", "yes")


def _as_download(resp):
    """Attach a marker cookie so the UI can clear its \"Preparing…\" state."""
    resp.set_cookie("sft_dl", "1", max_age=60, path="/")
    return resp

def _slice_profile_png(map3d, mask, ref, affine, measure):
    """Cohort band profile, or None if the reference predates it."""
    try:
        return _fig_to_b64(viz.plot_slice_profile(
            map3d, mask, ref, affine=affine,
            label=_measure_label(measure)))
    except ValueError:
        return None


def _hist_png(map3d, mask, exclude_zero=True, measure=None):
    label = _measure_label(measure) if measure else ""
    return _fig_to_b64(
        viz.plot_value_hist(map3d, mask, exclude_zero=exclude_zero, label=label)
    )


def _group_average(results):
    """Voxelwise mean map across subjects, ignoring out-of-mask voxels.

    Returns (mean_map, mask, affine) if the subjects share a grid, else None.
    """
    maps = [m for _, m, _, _ in results]
    masks = [mk for _, _, mk, _ in results]
    affine = results[0][3]
    if len(maps) < 2 or len({m.shape for m in maps}) != 1:
        return None
    stack = np.stack(maps)
    mstack = np.stack(masks)
    with np.errstate(invalid="ignore"):
        gmean = np.nanmean(np.where(mstack, stack, np.nan), axis=0)
    gmask = np.isfinite(gmean)
    return np.nan_to_num(gmean), gmask, affine


def _param_args(form):
    """Extract just the measure-parameter fields for a download URL."""
    keys = ("reho_cluster", "alff_tr", "alff_low", "alff_high")
    return {k: form.get(k) for k in keys if form is not None and form.get(k)}


def _download_url(measure, mode, sid, cached=None, folder=None, form=None):
    """Build a /download link that recomputes and returns the measure map."""
    q = {"measure": measure, "sid": sid, **_param_args(form)}
    if mode == "single" and cached:
        q["mode"] = "single"
        q["cached"] = cached
    elif mode in ("folder", "subject") and folder:
        q["mode"] = "folder"
        q["folder"] = folder
    else:
        return None
    return "/download?" + urlencode(q)


def _stats(map3d, mask):
    vals = map3d[mask] if mask is not None else map3d.ravel()
    return (
        f"{np.mean(vals):.3f}",
        f"{np.median(vals):.3f}",
        f"{int(mask.sum()) if mask is not None else vals.size:,}",
    )


def _compare_pngs(measure, map3d, mask, affine):
    """Comparison against the reference for this measure, if one exists.

    Returns a dict describing the comparison (or None if no reference):
      * mode "z"    -> z-scores vs a cohort with an SD map (percentile + z-map)
      * mode "diff" -> difference from the group average (mean-only reference)
      * mode "error"-> the reference could not be loaded
    """
    ref_path = measure_norm.default_path(measure)
    if not Path(ref_path).exists():
        return None
    try:
        ref = measure_norm.load(ref_path)
    except Exception as e:
        return {"mode": "error", "note": f"Could not load reference: {e}"}

    n = int(ref["n"]) if "n" in ref else 0

    # Standard space is not one grid: a subject can be perfectly good MNI data
    # at a different resolution. Resample the measure map onto the reference
    # grid rather than refusing the comparison.
    resampled = ""
    if affine is not None and measure_norm.needs_resampling(map3d, ref):
        src_shape = map3d.shape
        try:
            map3d, mask = measure_norm.resample_to_reference(
                map3d, mask, affine, ref)
            affine = measure_norm.reference_affine(ref)
            resampled = (
                f"Subject was on a {src_shape[0]}x{src_shape[1]}x{src_shape[2]} "
                f"grid and has been resampled to the cohort's "
                f"{'x'.join(str(int(s)) for s in ref['shape'])} grid for "
                "comparison. The measure itself was computed on the original "
                "grid."
            )
            if measure in ("reho", "coherence_reho"):
                resampled += (
                    " This measure summarises each voxel's neighbours, so its "
                    "values depend on voxel size; treat this comparison as "
                    "approximate."
                )
        except Exception as e:
            return {"mode": "error",
                    "note": f"Could not resample subject to the cohort grid: {e}"}

    # t-test path: one subject vs the group (needs SD and n > 1).
    if measure_norm.can_ttest(ref):
        try:
            t, p, tmask, df = measure_norm.ttest_vs_group(map3d, mask, ref)
        except ValueError as e:
            return {"mode": "error", "note": str(e)}
        tv = t[tmask]
        pv = p[tmask]
        n_sig = int((pv < 0.05).sum())
        # FDR (Benjamini-Hochberg) across tested voxels — the standard
        # multiple-comparison correction for voxelwise maps.
        q_vals, p_thr = measure_norm.fdr_correct(pv, alpha=0.05)
        n_fdr = int((q_vals < 0.05).sum())
        q_map = np.ones_like(p)
        q_map[tmask] = q_vals
        t_fdr = np.where(tmask & (q_map < 0.05), t, 0.0)
        # Mask provenance: warn if this subject was masked differently from the
        # cohort. Matters most for neighbourhood measures (ReHo, coherence-ReHo),
        # whose values depend on which neighbours are in the mask.
        ref_mask_name = str(ref["mask_name"]) if "mask_name" in ref else ""
        # Fall back to counting group_mask so references built before the
        # provenance fields existed still get the mismatch check.
        ref_mask_vox = (int(ref["mask_voxels"]) if "mask_voxels" in ref
                        else int(np.asarray(ref["group_mask"], bool).sum()))
        subj_vox = int(mask.sum())
        mask_warn = ""
        if ref_mask_vox and subj_vox != ref_mask_vox:
            nb = measure in ("reho", "coherence_reho")
            mask_warn = (
                f"This subject was analysed over {subj_vox:,} voxels but the "
                f"reference cohort used {ref_mask_vox:,}"
                + (f" ({ref_mask_name})" if ref_mask_name else "")
                + ". Only the shared voxels were tested."
                + (" Because this measure uses each voxel's neighbours, its "
                   "values also depend on the mask; upload the cohort mask for "
                   "an exact match." if nb else "")
            )
        # When only a handful of voxels survive, a brain figure is nearly blank
        # and easy to misread as a rendering failure. List the peaks instead.
        SPARSE_MAX = 20
        sparse = 0 < n_fdr <= SPARSE_MAX
        peaks = []
        if sparse:
            idx = np.argwhere(tmask & (q_map < 0.05))
            order = np.argsort(-np.abs(t[tuple(idx.T)]))
            for vox in idx[order]:
                mm = affine @ np.array([*vox, 1.0])
                peaks.append({
                    "xyz": f"{mm[0]:.0f}, {mm[1]:.0f}, {mm[2]:.0f}",
                    "t": f"{t[tuple(vox)]:+.2f}",
                    "q": f"{q_map[tuple(vox)]:.2e}",
                    "dir": "above" if t[tuple(vox)] > 0 else "below",
                })

        # Calibration check. Under the null ~5% of voxels land at p<0.05 and the
        # t values have SD~1. Far fewer, or an SD well under 1, means the group
        # SD is inflated — typically a raw-unit measure compared without global
        # normalisation, which shows up as a uniform whole-brain offset.
        t_sd = float(np.std(tv)) if tv.size else float("nan")
        obs_exp = (n_sig / (0.05 * tv.size)) if tv.size else float("nan")
        calib = ""
        if tv.size and (obs_exp < 0.5 or t_sd < 0.7):
            calib = (
                f"Only {n_sig:,} voxels reached p<0.05, versus "
                f"{int(0.05 * tv.size):,} expected by chance alone, and the t "
                f"values have SD {t_sd:.2f} where 1.0 is correctly calibrated. "
                "This points to an inflated cohort SD rather than a real result."
                + ("" if measure_norm.is_normalized(ref) else
                   " This reference was built from raw, un-normalised maps; for "
                   "amplitude measures (ALFF, RSFA) that lets between-subject "
                   "scanner scaling dominate the SD. Rebuild it with "
                   "scripts/reference_from_subject_maps.py.")
            )

        out = {
            "mode": "t", "n": n, "df": df, "note": "",
            "normalized": measure_norm.is_normalized(ref),
            "t_sd": f"{t_sd:.2f}" if tv.size else "n/a",
            "obs_exp": f"{obs_exp:.2f}" if tv.size else "n/a",
            "calib": calib,
            "resampled": resampled,
            "profile_png": _slice_profile_png(
                map3d, mask, ref, affine, measure),
            "sparse": sparse,
            "peaks": peaks,
            "ref_mask_name": ref_mask_name,
            "ref_mask_vox": f"{ref_mask_vox:,}" if ref_mask_vox else "",
            "mask_warn": mask_warn,
            "t_mean": f"{np.mean(tv):+.3f}" if tv.size else "n/a",
            "t_absmax": f"{np.max(np.abs(tv)):.2f}" if tv.size else "n/a",
            "n_tested": f"{int(tmask.sum()):,}",
            "n_sig": f"{n_sig:,}",
            "pct_sig": f"{100.0 * n_sig / tv.size:.1f}%" if tv.size else "n/a",
            "n_fdr": f"{n_fdr:,}",
            "pct_fdr": f"{100.0 * n_fdr / tv.size:.1f}%" if tv.size else "n/a",
            "p_thr": f"{p_thr:.2e}" if p_thr > 0 else "none survive",
            "t_png": _slices_png(
                t, affine, measure, symmetric=True, cmap="RdBu_r",
                title=f"{_measure_label(measure)}: t vs {n} HCP subjects "
                      f"(uncorrected; red = above group, blue = below)"),
            "t_fdr_png": _slices_png(
                t_fdr, affine, measure, symmetric=True, cmap="RdBu_r",
                title=f"{_measure_label(measure)}: t vs {n} HCP subjects "
                      f"(FDR q<0.05; {n_fdr:,} of {int(tmask.sum()):,} voxels survive)"),
            "map_png": None,
        }
        return out

    # z-score path (SD map but no usable n).
    if measure_norm.has_sd(ref):
        z, _zm, summary, pct = measure_norm.compare(map3d, mask, ref)
        out = {"mode": "z", "n": n, "note": "",
               "pct": "n/a" if pct != pct else f"{pct:.0f}th",
               "compare_png": None, "map_png": None}
        if np.asarray(ref["summaries"]).size:
            out["compare_png"] = _fig_to_b64(
                viz.plot_subject_vs_reference(summary, {"values": ref["summaries"]}))
        if z is not None:
            out["map_png"] = _slices_png(z, affine, measure, symmetric=True,
                                         cmap="RdBu_r")
        else:
            out["note"] = (
                f"Cohort grid {tuple(int(s) for s in ref['shape'])} differs from "
                f"this subject {map3d.shape}; showing summary percentile only.")
        return out

    # difference-from-average path (mean-only reference).
    diff, _dm, subj_mean, grp_mean = measure_norm.difference(map3d, mask, ref)
    finite = subj_mean == subj_mean and grp_mean == grp_mean
    out = {"mode": "diff", "n": n, "note": "",
           "subj_mean": "n/a" if subj_mean != subj_mean else f"{subj_mean:.3f}",
           "grp_mean": "n/a" if grp_mean != grp_mean else f"{grp_mean:.3f}",
           "diff_mean": f"{subj_mean - grp_mean:+.3f}" if finite else "n/a",
           "map_png": None}
    if diff is not None:
        out["map_png"] = _slices_png(diff, affine, measure, symmetric=True,
                                     cmap="RdBu_r")
    else:
        out["note"] = (
            f"Group grid {tuple(int(s) for s in ref['shape'])} differs from this "
            f"subject {map3d.shape}; showing summary means only.")
    return out


def _measure_block(measure, map3d, mask, affine, params_str, exclude_zero,
                   download_url=None, compare=True, stats_url=None,
                   mask_url=None):
    """Build one measure's result-card dict for the RESULT template."""
    mean, median, nvox = _stats(map3d, mask)
    cmp = _compare_pngs(measure, map3d, mask, affine) if compare else None
    return {
        "measure_key": measure,
        "measure_label": measures.MEASURES.get(measure, {}).get("label", measure),
        "params": params_str,
        "map_png": _slices_png(map3d, affine, measure),
        "hist_png": _hist_png(map3d, mask, exclude_zero=exclude_zero,
                              measure=measure),
        "mean": mean, "median": median, "nvox": nvox,
        "download_url": download_url,
        # t/p NIfTI export is only meaningful when the t-test actually ran.
        "stats_url": stats_url if (cmp and cmp.get("mode") == "t") else None,
        # The cohort mask is offered whenever a comparison ran, so users can
        # reproduce the analysis over exactly the voxels that were tested.
        "mask_url": mask_url if cmp else None,
        "cmp": cmp,
    }


def _render(sid, computed, notes, mode="single", folder=None, cached=None,
            cached_mask=None, form=None):
    """Render one or more measures for a single subject.

    ``computed`` is a list of (measure, map3d, mask, affine, params_str).
    """
    exclude_zero = not _truthy(form.get("include_zeros") if form is not None else None)
    blocks = []
    for measure, map3d, mask, affine, pstr in computed:
        dl = _download_url(measure, mode, sid, cached=cached, folder=folder, form=form)
        st = mk = None
        if mode == "single" and cached:
            st = "/download-stats?" + urlencode(
                dict(_param_args(form), measure=measure, sid=sid, cached=cached))
            mk = "/download-mask?" + urlencode(dict(measure=measure, cached=cached))
        blocks.append(_measure_block(measure, map3d, mask, affine, pstr,
                                     exclude_zero, download_url=dl, compare=True,
                                     stats_url=st, mask_url=mk))
    selected = [c[0] for c in computed]
    controls = _controls_html(mode, selected, folder=folder, cached=cached,
                              form=form, sid=sid, cached_mask=cached_mask)
    return render_template_string(RESULT, sid=sid, blocks=blocks, notes=notes,
                                  controls=controls)


def _render_batch(measures_list, results_by_measure, notes, folder=None,
                  form=None):
    """Render group-average results for a folder, one card per measure.

    Only group-level maps/stats are shown (no per-subject data), to keep the
    cohort de-identified. ``results_by_measure`` maps measure -> list of
    (sid, map3d, mask, affine).
    """
    exclude_zero = not _truthy(form.get("include_zeros") if form is not None else None)
    n_subj = max((len(r) for r in results_by_measure.values()), default=0)
    blocks = []
    for measure in measures_list:
        results = results_by_measure.get(measure, [])
        ga = _group_average(results)
        if ga is None:
            continue
        gmean, gmask, gaffine = ga
        base_q = dict(_param_args(form), measure=measure, folder=folder or "")
        block = _measure_block(measure, gmean, gmask, gaffine,
                               f"group average of {len(results)} subjects",
                               exclude_zero,
                               download_url="/download-group?" + urlencode(base_q),
                               compare=False)
        blocks.append(block)
    if not blocks:
        notes = (notes + " No group average could be formed (subjects need a "
                 "common grid, and at least two are required).").strip()
    sid = f"{n_subj} subjects · group average (de-identified)"
    controls = _controls_html("folder", measures_list, folder=folder, form=form)
    return render_template_string(RESULT, sid=sid, blocks=blocks, notes=notes,
                                  controls=controls)


# ---- routes -----------------------------------------------------------


@app.route("/")
def index():
    return render_template_string(PAGE, error=None, max_batch=MAX_BATCH)


def _page_error(msg):
    return render_template_string(PAGE, error=msg, max_batch=MAX_BATCH)


def _selected_measures(form):
    """Measures chosen via checkboxes (name=measures), or a single ?measure."""
    sel = [m for m in form.getlist("measures") if m in measures.MEASURES]
    if not sel:
        one = form.get("measure")
        if one in measures.MEASURES:
            sel = [one]
    # de-duplicate, preserve order
    seen, out = set(), []
    for m in sel:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


@app.route("/run", methods=["POST"])
def run():
    mode = request.form.get("mode", "single")
    selected = _selected_measures(request.form)
    if not selected:
        return _page_error("Select at least one measure.")

    if mode == "folder":
        folder = (request.form.get("folder") or "").strip()
        if not folder:
            return _page_error("Enter a subjects folder path.")
        try:
            bolds = io.find_subject_bolds(folder)
        except FileNotFoundError as e:
            return _page_error(str(e))
        if not bolds:
            return _page_error(f"No BOLD NIfTIs found under {folder}.")

        items = list(bolds.items())
        extra = len(items) - MAX_BATCH
        items = items[:MAX_BATCH]
        results_by_measure = {m: [] for m in selected}
        skipped = []
        for sid, path in items:
            try:
                img = io.load_nifti(path)
                bold = np.asarray(img.get_fdata())
                if bold.ndim != 4:
                    raise ValueError(f"not 4D (shape {bold.shape})")
            except Exception as e:
                skipped.append(f"{sid}: {e}")
                continue
            for measure in selected:
                try:
                    map3d, mask, _p, _c, _d = _compute(measure, bold, request.form)
                    results_by_measure[measure].append((sid, map3d, mask, img.affine))
                except Exception as e:
                    skipped.append(f"{sid}/{measure}: {e}")
        if not any(results_by_measure.values()):
            return _page_error("No subjects processed. " + "; ".join(skipped))
        notes = ""
        if extra > 0:
            notes += f"Showing first {MAX_BATCH}; {extra} more not processed. "
        if skipped:
            notes += f"Skipped {len(skipped)}: " + "; ".join(skipped[:10])
        return _render_batch(selected, results_by_measure, notes,
                             folder=folder, form=request.form)

    # Single-subject: use a cached upload if present (measure switch), else the
    # newly uploaded file, saved to the cache so later switches need no re-upload.
    sid = request.form.get("subject_id", "subject")
    cached = request.form.get("cached_bold")
    if cached:
        bold_p = CACHE_DIR / cached
        if not bold_p.exists():
            return _page_error("Cached data expired — please upload the BOLD again.")
    else:
        fs = request.files.get("bold")
        if fs is None or fs.filename == "":
            return _page_error("Cleaned BOLD is required.")
        cached = uuid.uuid4().hex + "_" + Path(fs.filename).name
        bold_p = CACHE_DIR / cached
        fs.save(str(bold_p))

    img = io.load_nifti(bold_p)
    bold = np.asarray(img.get_fdata())
    if bold.ndim != 4:
        return _page_error(f"BOLD must be 4D (X,Y,Z,T); got shape {bold.shape}.")

    # Optional analysis mask (uploaded once, then cached for measure switches).
    cached_mask = request.form.get("cached_mask") or ""
    mask_arr = None
    if cached_mask and (CACHE_DIR / cached_mask).exists():
        mask_arr = np.asarray(io.load_nifti_data(CACHE_DIR / cached_mask))
    else:
        mfs = request.files.get("mask")
        if mfs is not None and mfs.filename != "":
            cached_mask = uuid.uuid4().hex + "_" + Path(mfs.filename).name
            mfs.save(str(CACHE_DIR / cached_mask))
            mask_arr = np.asarray(io.load_nifti_data(CACHE_DIR / cached_mask))

    computed, skipped = [], []
    for measure in selected:
        try:
            map3d, mask, pstr, _c, _d = _compute(measure, bold, request.form,
                                                 mask=mask_arr)
            computed.append((measure, map3d, mask, img.affine, pstr))
        except Exception as e:
            skipped.append(f"{measure}: {e}")
    if not computed:
        return _page_error("No measures computed. " + "; ".join(skipped))
    notes = f"Skipped {'; '.join(skipped)}" if skipped else ""
    return _render(sid, computed, notes=notes, mode="single", cached=cached,
                   cached_mask=cached_mask, form=request.form)


_DEMO_BRAIN = None  # cached (mask, affine) so repeated demos don't reload


def _demo_brain():
    """A real MNI152 brain mask + affine for demos, at a coarse (fast) grid.

    Uses nilearn's bundled MNI152 brain mask (no download) at 4 mm so the demo
    slices look like an actual brain rather than a sphere, while staying small
    enough (~30k voxels) that ReHo/MSE compute in a few seconds. Falls back to
    downsampling the 2 mm mask on older nilearn versions.
    """
    global _DEMO_BRAIN
    if _DEMO_BRAIN is not None:
        return _DEMO_BRAIN
    from nilearn.datasets import load_mni152_brain_mask
    try:
        img = load_mni152_brain_mask(resolution=4)
        mask = np.asarray(img.get_fdata()) > 0
        affine = img.affine
    except TypeError:  # older nilearn without a resolution argument
        img = load_mni152_brain_mask()
        mask = (np.asarray(img.get_fdata()) > 0)[::2, ::2, ::2]
        affine = img.affine.copy()
        affine[:3, :3] *= 2
    _DEMO_BRAIN = (mask, affine)
    return _DEMO_BRAIN


def _synthetic_bold(T=100, seed=0):
    """Brain-shaped synthetic 4D BOLD for demos.

    Fills a real MNI152 brain mask with a handful of smooth spatial components,
    each carrying a low-frequency time course, plus noise. Spatially coherent
    signal (so ReHo/ALFF/etc. show real structure) inside an anatomically shaped
    brain, with background exactly zero. Returns (bold_4d, affine).
    """
    mask, affine = _demo_brain()
    X, Y, Z = mask.shape
    rng = np.random.default_rng(seed)
    a, b, c = np.mgrid[0:X, 0:Y, 0:Z].astype(np.float64)
    span = max(X, Y, Z)

    t = np.arange(T)
    signal = np.zeros((X, Y, Z, T), dtype=np.float64)
    for _ in range(8):
        cx, cy, cz = rng.uniform(0.28, 0.72, size=3) * np.array([X, Y, Z])
        sig = rng.uniform(0.08, 0.16) * span
        blob = np.exp(-(((a - cx) ** 2 + (b - cy) ** 2 + (c - cz) ** 2) / (2 * sig**2)))
        freq = rng.uniform(0.012, 0.09)
        phase = rng.uniform(0, 2 * np.pi)
        tc = np.sin(2 * np.pi * freq * t + phase)
        signal += blob[..., None] * tc[None, None, None, :]

    noise = rng.standard_normal((X, Y, Z, T)) * 0.25
    bold = (signal + noise) * mask[..., None]
    return bold, affine


@app.route("/subject", methods=["GET", "POST"])
def subject():
    """Individual subject views are withheld for folder cohorts.

    To keep a cohort de-identified, single-subject maps and per-subject stats
    are not exposed for folder uploads; only the group average is available.
    """
    return _page_error(
        "Individual subject views are disabled for folder cohorts. Only the "
        "group average is shown, to keep the cohort de-identified."
    )


@app.route("/download-batch")
def download_batch():
    """Withheld: per-subject maps are not downloadable for a folder cohort.

    Only the group average (see /download-group) is available, to keep the
    cohort de-identified.
    """
    abort(403)


@app.route("/download-group")
def download_group():
    """Voxelwise group-average measure map for a folder, as a NIfTI."""
    import nibabel as nib

    folder = request.args.get("folder", "")
    measure = request.args.get("measure", "reho")
    params = {
        "cluster": request.args.get("reho_cluster", 27),
        "tr": request.args.get("alff_tr", 0.72),
        "low": request.args.get("alff_low", 0.01),
        "high": request.args.get("alff_high", 0.08),
    }
    try:
        bolds = io.find_subject_bolds(folder)
    except Exception:
        abort(404)

    maps, masks, affine, shape = [], [], None, None
    for _sid, path in list(bolds.items())[:MAX_BATCH]:
        try:
            img = io.load_nifti(path)
            bold = np.asarray(img.get_fdata())
            if bold.ndim != 4:
                continue
            m, mask = measures.compute(measure, bold, params)
            if shape is None:
                shape, affine = m.shape, img.affine
            elif m.shape != shape:
                continue
            maps.append(m)
            masks.append(mask)
        except Exception:
            continue
    if len(maps) < 2:
        abort(400)

    stack, mstack = np.stack(maps), np.stack(masks)
    with np.errstate(invalid="ignore"):
        gmean = np.nan_to_num(np.nanmean(np.where(mstack, stack, np.nan), axis=0))
    out = CACHE_DIR / f"group_{measure}_average.nii.gz"
    nib.save(nib.Nifti1Image(gmean.astype(np.float32), affine), str(out))
    return _as_download(send_file(
        str(out), as_attachment=True, download_name=f"group_{measure}_average.nii.gz"
    ))


@app.route("/download-fa")
def download_fa():
    """Return the FA map with the chosen threshold applied, as a NIfTI."""
    import nibabel as nib

    cached_fa = request.args.get("cached_fa", "")
    thr = float(request.args.get("fa_threshold", 0.0) or 0.0)
    sid = request.args.get("sid", "subject")
    fa_p = CACHE_DIR / cached_fa
    if not cached_fa or not fa_p.exists():
        abort(404)
    img = io.load_nifti(fa_p)
    fa_map = np.asarray(img.get_fdata(), dtype=float)
    out_map = fa_mod.apply_threshold(fa_map, thr)
    out = CACHE_DIR / f"{sid}_FA_thr{thr:.2f}.nii.gz"
    nib.save(nib.Nifti1Image(out_map.astype(np.float32), img.affine), str(out))
    return _as_download(send_file(
        str(out), as_attachment=True,
        download_name=f"{sid}_FA_thr{thr:.2f}.nii.gz",
    ))


@app.route("/download-stats")
def download_stats():
    """Recompute a measure and return its t and p maps as NIfTI files (zipped).

    Only available for a single uploaded subject and a reference that supports
    the t-test (group SD + n > 1).
    """
    import zipfile

    import nibabel as nib

    measure = request.args.get("measure", "reho")
    sid = request.args.get("sid", "subject")
    cached = request.args.get("cached", "")
    path = CACHE_DIR / cached
    if not cached or not path.exists():
        abort(404)

    ref_path = measure_norm.default_path(measure)
    if not Path(ref_path).exists():
        abort(404)
    ref = measure_norm.load(ref_path)
    if not measure_norm.can_ttest(ref):
        abort(400)

    img = io.load_nifti(path)
    bold = np.asarray(img.get_fdata())
    params = {
        "cluster": request.args.get("reho_cluster", 27),
        "tr": request.args.get("alff_tr", 0.72),
        "low": request.args.get("alff_low", 0.01),
        "high": request.args.get("alff_high", 0.08),
    }
    try:
        m, mask = measures.compute(measure, bold, params)
        t, p, valid, _df = measure_norm.ttest_vs_group(m, mask, ref)
        q = np.ones_like(p)
        q[valid], _thr = measure_norm.fdr_correct(p[valid], alpha=0.05)
    except Exception:
        abort(400)

    zip_path = CACHE_DIR / f"{sid}_{measure}_stats.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, arr in (("tstat", t), ("pval", p), ("qval_fdr", q)):
            out = CACHE_DIR / f"{sid}_{measure}_{name}.nii.gz"
            nib.save(nib.Nifti1Image(arr.astype(np.float32), img.affine), str(out))
            zf.write(str(out), arcname=f"{sid}_{measure}_{name}.nii.gz")
        # Ship the cohort mask alongside: it defines which voxels were tested,
        # so results are not reproducible without it.
        gm = np.asarray(ref["group_mask"], bool)
        mout = CACHE_DIR / f"{measure}_cohort_mask.nii.gz"
        nib.save(nib.Nifti1Image(gm.astype(np.uint8), img.affine), str(mout))
        zf.write(str(mout), arcname="cohort_mask.nii.gz")
        zf.writestr("README.txt", _stats_readme(measure, ref, sid))
    return _as_download(send_file(
        str(zip_path), as_attachment=True,
        download_name=f"{sid}_{measure}_stats.zip",
    ))


def _stats_readme(measure: str, ref: dict, sid: str) -> str:
    """Plain-text provenance shipped inside the stats download."""
    label = measures.MEASURES.get(measure, {}).get("label", measure)
    n = int(ref["n"])
    mname = str(ref["mask_name"]) if "mask_name" in ref else ""
    return f"""Structure-Function Toolbox - single-subject statistics
=======================================================
Subject   : {sid}
Measure   : {label} ({measure})
Cohort    : n = {n} subjects, df = {n - 1}
Mask      : cohort_mask.nii.gz{f"  (built from {mname})" if mname else ""}
            {int(np.asarray(ref["group_mask"], bool).sum()):,} voxels

Files
-----
{sid}_{measure}_tstat.nii.gz     t statistic, per voxel
{sid}_{measure}_pval.nii.gz      two-tailed p, UNCORRECTED
{sid}_{measure}_qval_fdr.nii.gz  Benjamini-Hochberg FDR-adjusted p (q)
cohort_mask.nii.gz               voxels the test was run in (1 = tested)

Method
------
Crawford & Howell (1998) single-case t-test against the cohort:

    t = (subject - group_mean) / (group_SD * sqrt((n + 1) / n)),  df = n - 1

This treats the cohort as a finite sample rather than a known population, so
it is appropriate for comparing ONE subject to a normative group.

Reading the maps
----------------
Report the FDR map. Threshold qval_fdr at 0.05; the uncorrected p map is
provided for reference only and will contain many false positives at ~10^5
voxels. Values outside cohort_mask.nii.gz are not meaningful (t = 0, p = 1).

About the mask
--------------
The cohort mask is applied automatically, so the t-test is always confined to
these voxels. Supplying the same mask when you compute the measure is still
recommended: it keeps the reported summary statistics comparable, and for the
neighbourhood measures (ReHo, Coherence-ReHo) it changes the values themselves,
because those depend on which neighbouring voxels are included.
"""


@app.route("/download-mask")
def download_mask():
    """Return the cohort mask a reference was built with, as a NIfTI.

    The mask lives inside the reference, so this always matches the cohort the
    subject is being compared against — no separate file to keep in sync.
    """
    import nibabel as nib

    measure = request.args.get("measure", "reho")
    ref_path = measure_norm.default_path(measure)
    if not Path(ref_path).exists():
        abort(404)
    ref = measure_norm.load(ref_path)
    gm = np.asarray(ref["group_mask"], bool)

    # Prefer the subject's affine (same grid); else the standard MNI152 2mm one.
    affine = None
    cached = request.args.get("cached", "")
    if cached and (CACHE_DIR / cached).exists():
        try:
            affine = io.load_nifti(CACHE_DIR / cached).affine
        except Exception:
            affine = None
    if affine is None:
        affine = np.array([[-2., 0., 0., 90.], [0., 2., 0., -126.],
                           [0., 0., 2., -72.], [0., 0., 0., 1.]])

    out = CACHE_DIR / f"{measure}_cohort_mask.nii.gz"
    nib.save(nib.Nifti1Image(gm.astype(np.uint8), affine), str(out))
    return _as_download(send_file(
        str(out), as_attachment=True,
        download_name=f"{measure}_cohort_mask.nii.gz",
    ))


@app.route("/download")
def download():
    """Recompute the measure map for one subject and return it as a NIfTI."""
    import nibabel as nib

    measure = request.args.get("measure", "reho")
    mode = request.args.get("mode", "single")
    sid = request.args.get("sid", "subject")
    params = {
        "cluster": request.args.get("reho_cluster", 27),
        "tr": request.args.get("alff_tr", 0.72),
        "low": request.args.get("alff_low", 0.01),
        "high": request.args.get("alff_high", 0.08),
    }

    if mode == "folder":
        # Individual maps from a folder cohort are withheld (de-identification);
        # only the group average is downloadable.
        abort(403)
    else:
        cached = request.args.get("cached", "")
        path = CACHE_DIR / cached
        if not cached or not path.exists():
            abort(404)
        img = io.load_nifti(path)

    bold = np.asarray(img.get_fdata())
    try:
        m, _mask = measures.compute(measure, bold, params)
    except Exception:
        abort(400)

    out_img = nib.Nifti1Image(m.astype(np.float32), img.affine)
    out_path = CACHE_DIR / f"{sid}_{measure}.nii.gz"
    nib.save(out_img, str(out_path))
    return _as_download(send_file(
        str(out_path), as_attachment=True, download_name=f"{sid}_{measure}.nii.gz"
    ))


@app.route("/pick-folder")
def pick_folder():
    """Open a native folder chooser on the machine running the server.

    On macOS uses the system 'choose folder' dialog via osascript (smooth,
    native, no extra window). Falls back to a Tk dialog in a subprocess
    elsewhere.
    """
    import platform
    import subprocess
    import sys

    if platform.system() == "Darwin":
        script = (
            'POSIX path of (choose folder with prompt "Select the subjects folder")'
        )
        try:
            out = subprocess.run(
                ["osascript", "-e", script], capture_output=True, text=True, timeout=300
            )
            # Non-zero return with "User canceled" is a normal cancel.
            return {"path": out.stdout.strip()}
        except Exception as e:
            return {"path": "", "error": str(e)}

    code = (
        "import tkinter as tk;from tkinter import filedialog;"
        "r=tk.Tk();r.withdraw();r.attributes('-topmost',True);"
        "p=filedialog.askdirectory();print(p or '')"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
        )
        return {"path": out.stdout.strip()}
    except Exception as e:
        return {"path": "", "error": str(e)}


def _roi_bar_png(roi_values):
    """Bar chart of mean FA per parcel (NaN regions shown as gaps)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 2.8))
    x = np.arange(roi_values.size)
    ax.bar(x, np.nan_to_num(roi_values), color="#4a4a45", width=1.0)
    ax.set_xlabel("region")
    ax.set_ylabel("mean FA")
    ax.set_xlim(-0.5, roi_values.size - 0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _fig_to_b64(fig)


@app.route("/fa", methods=["POST"])
def fa_analyze():
    """Analyze an uploaded FA map at a chosen threshold.

    Applies the FA threshold, renders the thresholded FA map + histogram + basic
    stats, and (if an atlas is supplied) regional mean FA. The FA map and atlas
    are cached so the results-page slider can re-threshold without re-uploading.
    """
    thr = float(request.form.get("fa_threshold", 0.20) or 0.0)
    sid = request.form.get("sid") or "FA subject"

    # Reuse cached uploads on a slider re-threshold, else take fresh uploads.
    cached_fa = request.form.get("cached_fa") or ""
    cached_atlas = request.form.get("cached_atlas") or ""
    if cached_fa:
        fa_p = CACHE_DIR / cached_fa
        if not fa_p.exists():
            return _page_error("Cached FA map expired — please upload it again.")
    else:
        fs = request.files.get("fa")
        if fs is None or fs.filename == "":
            return _page_error("An FA map (3D NIfTI) is required.")
        cached_fa = uuid.uuid4().hex + "_" + Path(fs.filename).name
        fa_p = CACHE_DIR / cached_fa
        fs.save(str(fa_p))
        atlas_fs = request.files.get("atlas")
        if atlas_fs is not None and atlas_fs.filename != "":
            cached_atlas = uuid.uuid4().hex + "_" + Path(atlas_fs.filename).name
            atlas_fs.save(str(CACHE_DIR / cached_atlas))

    try:
        fa_img = io.load_nifti(fa_p)
        fa_map = np.asarray(fa_img.get_fdata(), dtype=float)
    except Exception as e:
        return _page_error(f"Could not read FA map: {e}")
    if fa_map.ndim != 3:
        return _page_error(f"FA map must be 3D (X,Y,Z); got shape {fa_map.shape}.")

    brain = fa_map > 0  # nonzero FA ~ within-brain support
    thr_map = fa_mod.apply_threshold(fa_map, thr)
    kept = thr_map > 0
    # Sequential map that starts light, so thresholded-out (zero) voxels read as
    # white background rather than the near-black low end of magma.
    map_png = _slices_png(thr_map, fa_img.affine, "fa", cmap="YlOrBr")
    hist_png = _hist_png(thr_map, kept)
    mean = f"{thr_map[kept].mean():.3f}" if kept.any() else "n/a"
    nvox = f"{int(kept.sum()):,}"
    pct_kept = f"{100.0 * kept.sum() / brain.sum():.1f}%" if brain.any() else "n/a"

    roi_png, roi_note, n_regions = None, "", 0
    if cached_atlas:
        try:
            labels = np.asarray(io.load_nifti_data(CACHE_DIR / cached_atlas))
            if labels.shape != fa_map.shape:
                roi_note = (
                    f"Atlas shape {labels.shape} does not match the FA "
                    f"map {fa_map.shape}; regional FA skipped."
                )
            else:
                roi = fa_mod.roi_fa(fa_map, labels, fa_threshold=thr)
                n_regions = int(roi.size)
                roi_png = _roi_bar_png(roi)
        except Exception as e:
            roi_note = f"Could not compute regional FA: {e}"

    return render_template_string(
        FA_RESULT,
        sid=sid,
        thr=f"{thr:.2f}",
        cached_fa=cached_fa,
        cached_atlas=cached_atlas,
        map_png=map_png,
        hist_png=hist_png,
        mean=mean,
        nvox=nvox,
        pct_kept=pct_kept,
        roi_png=roi_png,
        roi_note=roi_note,
        n_regions=n_regions,
        fa_dl_url=(
            "/download-fa?"
            + urlencode({"cached_fa": cached_fa, "fa_threshold": f"{thr}", "sid": sid})
            if cached_fa
            else None
        ),
        error=None,
    )


@app.route("/demo", methods=["POST"])
def demo():
    selected = _selected_measures(request.form)
    if not selected:
        return _page_error("Select at least one measure.")
    bold, affine = _synthetic_bold()
    note = "Synthetic BOLD in an MNI brain mask — demonstrates the layout, not real biology."
    computed, skipped = [], []
    for measure in selected:
        try:
            m, mask, pstr, _c, _d = _compute(measure, bold, request.form)
            computed.append((measure, m, mask, affine, pstr))
        except Exception as e:
            skipped.append(f"{measure}: {e}")
    if not computed:
        return _page_error("No measures computed. " + "; ".join(skipped))
    if skipped:
        note += "  Skipped " + "; ".join(skipped)
    return _render("DEMO (synthetic)", computed, notes=note, mode="demo",
                   form=request.form)


def main():
    import webbrowser

    url = "http://127.0.0.1:5000"
    print(f"Structure-Function Toolbox running at {url}  (Ctrl+C to stop)")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
