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

import numpy as np

# Headless plotting.
import matplotlib
matplotlib.use("Agg")

from flask import Flask, request, render_template_string, send_file, abort

from . import io, measures, measure_norm, viz


app = Flask(__name__)

# Persistent per-session cache so a single-subject upload can be reused when
# the user switches measures on the results page (no re-upload needed).
CACHE_DIR = Path(tempfile.mkdtemp(prefix="sft_cache_"))

# ---- HTML (kept inline so the app is one file) ------------------------

STYLE = """
<style>
 :root{
   --bg:#f4f8fc; --surface:#ffffff; --border:#e2ebf3; --border-strong:#cdddea;
   --text:#1e293b; --muted:#64748b; --heading:#0f2740;
   --accent:#3b82c4; --accent-hover:#2f6aa3; --accent-soft:#e6f0f9;
   --secondary:#eaf2f9; --secondary-hover:#dae7f2; --secondary-text:#3e5266;
   --err:#b3492f; --err-soft:#fbeae5;
   --shadow:0 1px 2px rgba(15,39,64,.04),0 4px 12px rgba(15,39,64,.05);
   --shadow-hover:0 2px 4px rgba(15,39,64,.06),0 8px 20px rgba(15,39,64,.10);
 }
 *{box-sizing:border-box;}
 body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;
      margin:0;background:var(--bg);color:var(--text);line-height:1.55;
      -webkit-font-smoothing:antialiased;}
 header{background:var(--surface);border-bottom:1px solid var(--border);
        padding:1rem 1.5rem;position:sticky;top:0;z-index:5;}
 header .wrap{max-width:820px;margin:0 auto;display:flex;align-items:baseline;
              gap:.75rem;}
 header .title{font-size:1.05rem;font-weight:700;color:var(--heading);
               letter-spacing:-.01em;}
 header .tag{font-size:.8rem;color:var(--muted);}
 main{max-width:820px;margin:0 auto;padding:1.75rem 1.5rem 3rem;}
 main.wide{max-width:1180px;}
 .lead{color:var(--muted);font-size:.95rem;margin:.25rem 0 1.5rem;max-width:60ch;}
 h2{font-size:1.05rem;font-weight:650;color:var(--heading);
    margin:0 0 .25rem;letter-spacing:-.01em;}
 .card{background:var(--surface);border:1px solid var(--border);
       border-radius:14px;padding:1.4rem 1.5rem;margin:1.1rem 0;
       box-shadow:var(--shadow);}
 .card .sub{color:var(--muted);font-size:.85rem;margin:.1rem 0 1rem;}
 label{display:block;margin:.9rem 0 .35rem;font-weight:600;font-size:.82rem;
        color:var(--heading);letter-spacing:.005em;}
 input[type=text],input[type=number],input[type=file],select{width:100%;
        padding:.6rem .7rem;border:1px solid var(--border-strong);
        border-radius:9px;background:#fff;font-size:.9rem;color:var(--text);
        transition:border-color .15s ease,box-shadow .15s ease;}
 select{appearance:none;-webkit-appearance:none;cursor:pointer;
        background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath fill='%233b82c4' d='M1 1l5 5 5-5'/%3E%3C/svg%3E");
        background-repeat:no-repeat;background-position:right .8rem center;
        padding-right:2rem;}
 input:focus,select:focus{outline:none;border-color:var(--accent);
        box-shadow:0 0 0 3px var(--accent-soft);}
 input[type=file]{padding:.45rem .5rem;background:#fdfcfa;cursor:pointer;}
 .row{display:flex;gap:1rem;flex-wrap:wrap;} .row>div{flex:1;min-width:150px;}
 .params{border-top:1px solid var(--border);margin-top:1.2rem;padding-top:.4rem;}
 .params .phint{color:var(--muted);font-size:.78rem;margin-top:.6rem;}
 .phint{color:var(--muted);font-size:.78rem;margin-top:.6rem;}
 .pgroup{display:none;} .pgroup.active{display:block;}
 .mgroup{display:none;} .mgroup.active{display:block;}
 .pickrow{display:flex;gap:.6rem;align-items:stretch;}
 .pickrow input{flex:1;} .pickrow button{white-space:nowrap;}
 .demorow{display:flex;gap:.7rem;flex-wrap:wrap;margin-top:1rem;}
 .actions{margin-top:1.5rem;display:flex;gap:.7rem;flex-wrap:wrap;}
 button{font-family:inherit;font-size:.92rem;font-weight:600;cursor:pointer;
        border-radius:10px;padding:.7rem 1.35rem;border:1px solid transparent;
        transition:background .15s ease,box-shadow .15s ease,transform .06s ease;}
 button:active{transform:translateY(1px);}
 .btn-primary{background:var(--accent);color:#fff;box-shadow:var(--shadow);}
 .btn-primary:hover{background:var(--accent-hover);box-shadow:var(--shadow-hover);}
 .btn-secondary{background:var(--secondary);color:var(--secondary-text);
        border-color:var(--border-strong);}
 .btn-secondary:hover{background:var(--secondary-hover);}
 a.btn-secondary,a.btn-primary{display:inline-block;text-decoration:none;
        font-weight:600;font-size:.9rem;border-radius:10px;padding:.6rem 1.1rem;
        border:1px solid transparent;transition:background .15s ease;}
 a.btn-secondary{background:var(--secondary);color:var(--secondary-text);
        border-color:var(--border-strong);}
 a.btn-secondary:hover{background:var(--secondary-hover);text-decoration:none;}
 .dllink{font-size:.82rem;}
 .muted{color:var(--muted);font-size:.85rem;}
 .hint{color:var(--muted);font-size:.8rem;margin-top:.9rem;line-height:1.5;}
 img{max-width:100%;border-radius:10px;display:block;}
 a{color:var(--accent);text-decoration:none;font-weight:600;}
 a:hover{text-decoration:underline;}
 .back{display:inline-block;margin-bottom:.5rem;font-size:.88rem;}
 .stats{display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:.4rem;}
 .stat .k{font-size:.72rem;color:var(--muted);text-transform:uppercase;
          letter-spacing:.04em;}
 .stat .v{font-size:1.25rem;font-weight:700;color:var(--accent);}
 .err{background:var(--err-soft);border-color:#f0cabd;color:var(--err);}
 .err b{color:var(--err);}
</style>
"""

PAGE = """
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Structure-Function Toolbox</title>
""" + STYLE + """
</head><body>
<header><div class="wrap">
  <span class="title">Structure-Function Toolbox</span>
  <span class="tag">local &middot; fMRI measures</span>
</div></header>
<main>
<p class="lead">Upload a subject's cleaned BOLD, choose an fMRI measure, set its
parameters, and generate the map. Everything runs on your machine.</p>

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
    <h2 style="margin-top:.6rem;">2 &nbsp;fMRI measure</h2>
    <label>Measure</label>
    <select name="measure" id="measure" onchange="showParams()">
      <option value="reho">Regional Homogeneity (ReHo)</option>
      <option value="alff">ALFF</option>
      <option value="falff">fALFF (fractional ALFF)</option>
      <option value="rsfa">RSFA (Resting-State Fluctuation Amplitude)</option>
    </select>

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

    <!-- ALFF / fALFF params (shared) -->
    <div class="pgroup" data-measure="alff falff">
      <div class="row">
        <div><label>TR (seconds)</label>
             <input type="number" name="alff_tr" step="0.01" value="2.0"></div>
        <div><label>Low (Hz)</label>
             <input type="number" name="alff_low" step="0.001" value="0.01"></div>
        <div><label>High (Hz)</label>
             <input type="number" name="alff_high" step="0.001" value="0.08"></div>
      </div>
      <p class="phint">ALFF = mean amplitude in band; fALFF = band / total.
      TR sets the frequency axis.</p>
    </div>

    <!-- RSFA has no parameters -->
    <div class="pgroup" data-measure="rsfa">
      <p class="phint">Temporal standard deviation of each voxel's BOLD signal.
      No parameters.</p>
    </div>
  </div>

  <div class="actions">
    <button type="submit" class="btn-primary">Generate measure</button>
  </div>
</form>

<form method="post" action="/demo" class="card">
  <h2>No data handy?</h2>
  <p class="sub">Run any measure on synthetic BOLD to preview its output.</p>
  <div class="demorow">
    <button type="submit" name="measure" value="reho" class="btn-secondary">Demo: ReHo</button>
    <button type="submit" name="measure" value="alff" class="btn-secondary">Demo: ALFF</button>
    <button type="submit" name="measure" value="falff" class="btn-secondary">Demo: fALFF</button>
    <button type="submit" name="measure" value="rsfa" class="btn-secondary">Demo: RSFA</button>
  </div>
</form>

<script>
function pickFolder(){
  fetch('/pick-folder').then(function(r){return r.json();}).then(function(d){
    if(d.path){ document.getElementById('folder').value = d.path; }
    else if(d.error){ alert('Folder picker unavailable: ' + d.error +
      '\\nYou can paste the path manually.'); }
  }).catch(function(e){ alert('Could not open picker: ' + e); });
}
function showParams(){
  var m = document.getElementById('measure').value;
  document.querySelectorAll('.pgroup').forEach(function(g){
    var ms = g.getAttribute('data-measure').split(' ');
    g.classList.toggle('active', ms.indexOf(m) !== -1);
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
</body></html>
"""

RESULT = """
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Results — {{ sid }}</title>
""" + STYLE + """
</head><body>
<header><div class="wrap">
  <span class="title">Structure-Function Toolbox</span>
  <span class="tag">results</span>
</div></header>
<main>
<a class="back" href="/">&larr; New analysis</a>
<h2 style="font-size:1.25rem;margin:.3rem 0 .2rem;">{{ sid }}</h2>
<p class="muted">{{ measure_label }}{% if params %} &middot; {{ params }}{% endif %}</p>

<div class="card">
  <h2>{{ measure_label }}</h2>
  <p class="sub">Axial, coronal, and sagittal slices through the measure map.</p>
  <img src="data:image/png;base64,{{ map_png }}">
  <div class="stats">
    <div class="stat"><div class="k">mean</div><div class="v">{{ mean }}</div></div>
    <div class="stat"><div class="k">median</div><div class="v">{{ median }}</div></div>
    <div class="stat"><div class="k">masked voxels</div><div class="v">{{ nvox }}</div></div>
  </div>
  <img src="data:image/png;base64,{{ hist_png }}" style="margin-top:1rem;">
  <div class="actions">
    {% if download_url %}
    <a class="btn-secondary" href="{{ download_url }}">Download map (.nii.gz)</a>
    {% endif %}
    <a class="btn-secondary" download="{{ sid }}_{{ measure_key }}.png"
       href="data:image/png;base64,{{ map_png }}">Download image (PNG)</a>
  </div>
</div>

{% if compare_png %}
<div class="card">
  <h2>Subject vs. cohort</h2>
  <p class="sub">Where this subject's mean {{ measure_label }} falls among
  {{ comp_n }} reference subjects.</p>
  <div class="stats" style="margin-bottom:.8rem;">
    <div class="stat"><div class="k">percentile</div><div class="v">{{ comp_pct }}</div></div>
  </div>
  <img src="data:image/png;base64,{{ compare_png }}">
</div>
{% endif %}

{% if zmap_png %}
<div class="card">
  <h2>Deviation z-map</h2>
  <p class="sub">Voxelwise (subject &minus; cohort mean) / cohort SD. Red =
  above the cohort, blue = below.</p>
  <img src="data:image/png;base64,{{ zmap_png }}">
</div>
{% endif %}

{% if comp_note %}<div class="card muted">{{ comp_note }}</div>{% endif %}
{% if notes %}<div class="card muted">{{ notes }}</div>{% endif %}

{{ controls|safe }}
</main>
</body></html>
"""

BATCH = """
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Batch results</title>
""" + STYLE + """
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
 .sidebar a{display:block;padding:.35rem .5rem;border-radius:7px;font-size:.85rem;
            font-weight:500;color:var(--secondary-text);text-decoration:none;
            white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
 .sidebar a:hover{background:var(--secondary);}
 .sidebar a.top{color:var(--accent);font-weight:600;margin-bottom:.3rem;}
 .maincol{flex:1;min-width:0;}
 .rightbar{position:sticky;top:66px;flex:0 0 235px;
           max-height:calc(100vh - 88px);overflow:auto;}
 .rightbar .card{margin-top:0;}
 .subject-card{scroll-margin-top:74px;}
 #backtop{position:fixed;bottom:1.5rem;right:1.5rem;z-index:30;display:none;
          box-shadow:var(--shadow-hover);}
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
    <h3>Subjects</h3>
    {% for r in rows %}
    <a href="#subj{{ loop.index0 }}">{{ r.sid }}</a>
    {% endfor %}
  </nav>

  <div class="maincol">
    <div class="card">
      <h2>Summary</h2>
      <p class="sub">Per-subject mean / median across the brain mask. Click a
      subject to jump to it.</p>
      <table>
        <tr><th>Subject</th><th>Mean</th><th>Median</th><th>Masked voxels</th>
            {% if has_pct %}<th>Cohort pctile</th>{% endif %}<th></th></tr>
        {% for r in rows %}
        <tr><td>{{ r.sid }}</td><td class="num">{{ r.mean }}</td>
            <td class="num">{{ r.median }}</td><td class="num">{{ r.nvox }}</td>
            {% if has_pct %}<td class="num">{{ r.pct }}</td>{% endif %}
            <td class="viewcell"><a href="#subj{{ loop.index0 }}">View &darr;</a></td></tr>
        {% endfor %}
      </table>
      <div class="actions">
        <a class="btn-primary" href="{{ download_all_url }}">Download all maps (.zip)</a>
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

    {% for r in rows %}
    <div class="card subject-card" id="subj{{ loop.index0 }}">
      <h2>{{ r.sid }}</h2>
      <img src="data:image/png;base64,{{ r.png }}">
      <div class="actions">
        <a class="btn-secondary" href="{{ r.download_url }}">Download map (.nii.gz)</a>
        <a class="btn-secondary" download="{{ r.sid }}_{{ measure_key }}.png"
           href="data:image/png;base64,{{ r.png }}">Download image (PNG)</a>
      </div>
    </div>
    {% endfor %}

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


def _compute(measure: str, bold: np.ndarray, form):
    """Dispatch to the chosen measure; return (map3d, mask, params_str)."""
    if measure == "reho":
        cl = int(form.get("reho_cluster", 27))
        m, mask = measures.reho(bold, cluster=cl)
        return m, mask, f"cluster {cl}", "magma", False
    if measure in ("alff", "falff"):
        tr = float(form.get("alff_tr", 2.0))
        lo = float(form.get("alff_low", 0.01))
        hi = float(form.get("alff_high", 0.08))
        alff, falff, mask = measures.alff_falff(bold, tr=tr, band=(lo, hi))
        out = alff if measure == "alff" else falff
        return out, mask, f"TR {tr}s, band {lo}-{hi} Hz", "magma", False
    if measure == "rsfa":
        m, mask = measures.rsfa(bold)
        return m, mask, "temporal std of BOLD", "magma", False
    raise ValueError(f"unknown measure: {measure}")


def _controls_html(mode, measure, folder=None, cached=None, form=None, sid=None):
    """A compact 'change measure & regenerate' form for the results pages.

    Carries the datasource (folder path, cached upload, or folder+subject) as
    hidden fields so the user never re-selects it. In demo mode it re-runs /demo.
    """
    g = (form.get if form is not None else (lambda k, d=None: d))
    reho_c = g("reho_cluster", "27")
    tr = g("alff_tr", "2.0"); lo = g("alff_low", "0.01"); hi = g("alff_high", "0.08")

    action = {"demo": "/demo", "subject": "/subject"}.get(mode, "/run")
    hidden = f'<input type="hidden" name="mode" value="{mode}">'
    if mode in ("folder", "subject") and folder:
        hidden += f'<input type="hidden" name="folder" value="{html.escape(folder)}">'
    if mode == "subject" and sid:
        hidden += f'<input type="hidden" name="sid" value="{html.escape(sid)}">'
    if mode == "single" and cached:
        hidden += f'<input type="hidden" name="cached_bold" value="{html.escape(cached)}">'
    same = {"demo": "same synthetic data", "folder": "same folder",
            "subject": "same subject"}.get(mode, "same subject")

    def opt(v, label):
        return f'<option value="{v}"{" selected" if v == measure else ""}>{label}</option>'

    def rsel(v):
        return " selected" if reho_c == v else ""

    options = (opt("reho", "Regional Homogeneity (ReHo)") + opt("alff", "ALFF")
               + opt("falff", "fALFF (fractional ALFF)")
               + opt("rsfa", "RSFA (Resting-State Fluctuation Amplitude)"))

    return f'''
<form method="post" action="{action}" class="card">
  <h2>Change measure</h2>
  <p class="sub">Switch measure or adjust parameters and regenerate &mdash; {same}, no re-selecting.</p>
  {hidden}
  <label>Measure</label>
  <select name="measure" id="measure" onchange="showParams()">{options}</select>

  <div class="pgroup" data-measure="reho">
    <label>Cluster size</label>
    <select name="reho_cluster">
      <option value="7"{rsel("7")}>7 voxels (faces)</option>
      <option value="19"{rsel("19")}>19 voxels (+ edges)</option>
      <option value="27"{rsel("27")}>27 voxels (+ corners)</option>
    </select>
  </div>
  <div class="pgroup" data-measure="alff falff">
    <div class="row">
      <div><label>TR (s)</label><input type="number" name="alff_tr" step="0.01" value="{tr}"></div>
      <div><label>Low (Hz)</label><input type="number" name="alff_low" step="0.001" value="{lo}"></div>
      <div><label>High (Hz)</label><input type="number" name="alff_high" step="0.001" value="{hi}"></div>
    </div>
  </div>
  <div class="pgroup" data-measure="rsfa">
    <p class="phint">Temporal standard deviation of BOLD. No parameters.</p>
  </div>

  <div class="actions"><button class="btn-primary">Regenerate</button></div>
  <script>
  function showParams(){{
    var m=document.getElementById('measure').value;
    document.querySelectorAll('.pgroup').forEach(function(g){{
      var ms=g.getAttribute('data-measure').split(' ');
      g.classList.toggle('active', ms.indexOf(m)!==-1);
    }});
  }}
  showParams();
  </script>
</form>'''


def _slices_png(map3d, affine, measure, symmetric=False, cmap=None):
    cmap = cmap or viz._MEASURE_CMAP.get(measure, "cold_hot")
    return _fig_to_b64(viz.plot_orientations(
        map3d, affine, cmap=cmap, symmetric=symmetric))


def _hist_png(map3d, mask):
    return _fig_to_b64(viz.plot_value_hist(map3d, mask))


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
        q["mode"] = "single"; q["cached"] = cached
    elif mode in ("folder", "subject") and folder:
        q["mode"] = "folder"; q["folder"] = folder
    else:
        return None
    return "/download?" + urlencode(q)


def _stats(map3d, mask):
    vals = map3d[mask] if mask is not None else map3d.ravel()
    return (f"{np.mean(vals):.3f}", f"{np.median(vals):.3f}",
            f"{int(mask.sum()) if mask is not None else vals.size:,}")


def _compare_pngs(measure, map3d, mask, affine):
    """If a reference for this measure exists, return comparison figures.

    Returns (compare_png, zmap_png, pct_str, n, note).
    """
    ref_path = measure_norm.default_path(measure)
    if not Path(ref_path).exists():
        return None, None, None, None, ""
    try:
        ref = measure_norm.load(ref_path)
        z, z_mask, summary, pct = measure_norm.compare(map3d, mask, ref)
    except Exception as e:
        return None, None, None, None, f"Could not load reference: {e}"
    compare_png = _fig_to_b64(viz.plot_subject_vs_reference(
        summary, {"values": ref["summaries"]}))
    zmap_png, note = None, ""
    if z is not None:
        zmap_png = _slices_png(z, affine, measure, symmetric=True, cmap="cold_hot")
    else:
        note = (f"Cohort grid {tuple(int(s) for s in ref['shape'])} differs from "
                f"this subject {map3d.shape}; showing summary percentile only "
                "(resample to a common space for a voxelwise z-map).")
    pct_str = "n/a" if pct != pct else f"{pct:.0f}th"
    return compare_png, zmap_png, pct_str, int(ref["n"]), note


def _render(sid, measure, map3d, mask, params, notes, affine,
            mode="single", folder=None, cached=None, form=None):
    label = measures.MEASURES.get(measure, {}).get("label", measure)
    png = _slices_png(map3d, affine, measure)
    hist_png = _hist_png(map3d, mask)
    mean, median, nvox = _stats(map3d, mask)
    compare_png, zmap_png, comp_pct, comp_n, comp_note = _compare_pngs(
        measure, map3d, mask, affine)
    controls = _controls_html(mode, measure, folder=folder, cached=cached,
                              form=form, sid=sid)
    download_url = _download_url(measure, mode, sid, cached=cached,
                                 folder=folder, form=form)
    return render_template_string(
        RESULT, sid=sid, measure_label=label, measure_key=measure, params=params,
        map_png=png, hist_png=hist_png, mean=mean, median=median, nvox=nvox,
        notes=notes, controls=controls, compare_png=compare_png,
        zmap_png=zmap_png, comp_pct=comp_pct, comp_n=comp_n, comp_note=comp_note,
        download_url=download_url)


def _render_batch(measure, results, params, notes, folder=None, form=None):
    label = measures.MEASURES.get(measure, {}).get("label", measure)
    ref_path = measure_norm.default_path(measure)
    try:
        ref = measure_norm.load(ref_path) if Path(ref_path).exists() else None
    except Exception:
        ref = None
    view_q = dict(_param_args(form), measure=measure, folder=folder or "")
    rows = []
    for sid, map3d, mask, affine in results:
        mean, median, nvox = _stats(map3d, mask)
        row = {"sid": sid, "mean": mean, "median": median, "nvox": nvox,
               "png": _slices_png(map3d, affine, measure), "pct": None,
               "download_url": _download_url(measure, "folder", sid,
                                             folder=folder, form=form),
               "view_url": "/subject?" + urlencode(dict(view_q, sid=sid))}
        if ref is not None:
            _z, _zm, _s, pct = measure_norm.compare(map3d, mask, ref)
            row["pct"] = "n/a" if pct != pct else f"{pct:.0f}th"
        rows.append(row)
    controls = _controls_html("folder", measure, folder=folder, form=form)
    base_q = dict(_param_args(form), measure=measure, folder=folder or "")
    download_all_url = "/download-batch?" + urlencode(base_q)

    # Group-average map (voxelwise mean across subjects on a common grid).
    group_png = group_hist_png = group_dl_url = None
    group_note = ""
    ga = _group_average(results)
    if ga is not None:
        gmean, gmask, gaffine = ga
        group_png = _slices_png(gmean, gaffine, measure)
        group_hist_png = _hist_png(gmean, gmask)
        group_dl_url = "/download-group?" + urlencode(base_q)
    elif len(results) >= 2:
        group_note = ("A voxelwise group average needs all subjects on the same "
                      "grid/space; these subjects differ, so no average is shown.")

    return render_template_string(
        BATCH, n=len(rows), measure_label=label, measure_key=measure,
        params=params, rows=rows, notes=notes, controls=controls,
        has_pct=ref is not None, download_all_url=download_all_url,
        group_png=group_png, group_hist_png=group_hist_png,
        group_dl_url=group_dl_url, group_note=group_note)


# ---- routes -----------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(PAGE, error=None, max_batch=MAX_BATCH)


def _page_error(msg):
    return render_template_string(PAGE, error=msg, max_batch=MAX_BATCH)


@app.route("/run", methods=["POST"])
def run():
    measure = request.form.get("measure", "reho")
    mode = request.form.get("mode", "single")

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
        results, skipped = [], []
        for sid, path in items:
            try:
                img = io.load_nifti(path)
                bold = np.asarray(img.get_fdata())
                if bold.ndim != 4:
                    raise ValueError(f"not 4D (shape {bold.shape})")
                map3d, mask, params, _c, _d = _compute(measure, bold, request.form)
                results.append((sid, map3d, mask, img.affine))
            except Exception as e:
                skipped.append(f"{sid}: {e}")
        if not results:
            return _page_error("No subjects processed. " + "; ".join(skipped))
        notes = ""
        if extra > 0:
            notes += f"Showing first {MAX_BATCH}; {extra} more not processed. "
        if skipped:
            notes += f"Skipped {len(skipped)}: " + "; ".join(skipped)
        return _render_batch(measure, results, params, notes,
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
    try:
        map3d, mask, params, _c, _d = _compute(measure, bold, request.form)
    except Exception as e:
        return _page_error(str(e))
    return _render(sid, measure, map3d, mask, params, notes="", affine=img.affine,
                   mode="single", cached=cached, form=request.form)


def _synthetic_bold(dim=28, T=120, seed=0):
    """Brain-like synthetic 4D BOLD for demos.

    A spherical "brain" filled with a handful of smooth spatial components, each
    carrying a low-frequency time course. This gives spatially coherent signal
    (so ReHo, ALFF, and seed FC show real structure) instead of salt-and-pepper
    noise, and a non-flat value distribution.
    """
    rng = np.random.default_rng(seed)
    a, b, c = np.mgrid[0:dim, 0:dim, 0:dim]
    ctr = (dim - 1) / 2
    radius = dim * 0.42
    dist = np.sqrt((a - ctr) ** 2 + (b - ctr) ** 2 + (c - ctr) ** 2)
    mask = dist <= radius

    t = np.arange(T)
    signal = np.zeros((dim, dim, dim, T), dtype=np.float64)
    for _ in range(6):
        cx, cy, cz = rng.uniform(dim * 0.30, dim * 0.70, size=3)
        sig = rng.uniform(dim * 0.10, dim * 0.20)
        blob = np.exp(-(((a - cx) ** 2 + (b - cy) ** 2 + (c - cz) ** 2)
                        / (2 * sig ** 2)))
        freq = rng.uniform(0.012, 0.09)
        phase = rng.uniform(0, 2 * np.pi)
        tc = np.sin(2 * np.pi * freq * t + phase)
        signal += blob[..., None] * tc[None, None, None, :]

    noise = rng.standard_normal((dim, dim, dim, T)) * 0.25
    bold = (signal + noise) * mask[..., None]
    return bold


@app.route("/subject", methods=["GET", "POST"])
def subject():
    """Full single-subject view for one subject inside a folder upload."""
    src = request.values
    folder = src.get("folder", "")
    sid = src.get("sid", "")
    measure = src.get("measure", "reho")
    try:
        bolds = io.find_subject_bolds(folder)
    except Exception:
        return _page_error("Folder not found.")
    path = bolds.get(sid)
    if path is None:
        return _page_error(f"Subject '{sid}' not found in {folder}.")
    img = io.load_nifti(path)
    bold = np.asarray(img.get_fdata())
    if bold.ndim != 4:
        return _page_error(f"BOLD must be 4D; got {bold.shape}.")
    try:
        map3d, mask, params, _c, _d = _compute(measure, bold, src)
    except Exception as e:
        return _page_error(str(e))
    return _render(sid, measure, map3d, mask, params, notes="",
                   affine=img.affine, mode="subject", folder=folder, form=src)


@app.route("/download-batch")
def download_batch():
    """Zip of every subject's measure map for a folder + measure."""
    import zipfile
    import nibabel as nib

    folder = request.args.get("folder", "")
    measure = request.args.get("measure", "reho")
    params = {
        "cluster": request.args.get("reho_cluster", 27),
        "tr": request.args.get("alff_tr", 2.0),
        "low": request.args.get("alff_low", 0.01),
        "high": request.args.get("alff_high", 0.08),
    }
    try:
        bolds = io.find_subject_bolds(folder)
    except Exception:
        abort(404)
    if not bolds:
        abort(404)

    zip_path = CACHE_DIR / f"{measure}_maps.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for sid, path in list(bolds.items())[:MAX_BATCH]:
            try:
                img = io.load_nifti(path)
                bold = np.asarray(img.get_fdata())
                if bold.ndim != 4:
                    continue
                m, _mask = measures.compute(measure, bold, params)
                out = CACHE_DIR / f"{sid}_{measure}.nii.gz"
                nib.save(nib.Nifti1Image(m.astype(np.float32), img.affine), str(out))
                zf.write(str(out), arcname=f"{sid}_{measure}.nii.gz")
            except Exception:
                continue
    return send_file(str(zip_path), as_attachment=True,
                     download_name=f"{measure}_maps.zip")


@app.route("/download-group")
def download_group():
    """Voxelwise group-average measure map for a folder, as a NIfTI."""
    import nibabel as nib

    folder = request.args.get("folder", "")
    measure = request.args.get("measure", "reho")
    params = {
        "cluster": request.args.get("reho_cluster", 27),
        "tr": request.args.get("alff_tr", 2.0),
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
            maps.append(m); masks.append(mask)
        except Exception:
            continue
    if len(maps) < 2:
        abort(400)

    stack, mstack = np.stack(maps), np.stack(masks)
    with np.errstate(invalid="ignore"):
        gmean = np.nan_to_num(np.nanmean(np.where(mstack, stack, np.nan), axis=0))
    out = CACHE_DIR / f"group_{measure}_average.nii.gz"
    nib.save(nib.Nifti1Image(gmean.astype(np.float32), affine), str(out))
    return send_file(str(out), as_attachment=True,
                     download_name=f"group_{measure}_average.nii.gz")


@app.route("/download")
def download():
    """Recompute the measure map for one subject and return it as a NIfTI."""
    import nibabel as nib

    measure = request.args.get("measure", "reho")
    mode = request.args.get("mode", "single")
    sid = request.args.get("sid", "subject")
    params = {
        "cluster": request.args.get("reho_cluster", 27),
        "tr": request.args.get("alff_tr", 2.0),
        "low": request.args.get("alff_low", 0.01),
        "high": request.args.get("alff_high", 0.08),
    }

    if mode == "folder":
        folder = request.args.get("folder", "")
        try:
            bolds = io.find_subject_bolds(folder)
        except Exception:
            abort(404)
        path = bolds.get(sid)
        if path is None:
            abort(404)
        img = io.load_nifti(path)
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
    return send_file(str(out_path), as_attachment=True,
                     download_name=f"{sid}_{measure}.nii.gz")


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
        script = ('POSIX path of (choose folder with prompt '
                  '"Select the subjects folder")')
        try:
            out = subprocess.run(["osascript", "-e", script],
                                 capture_output=True, text=True, timeout=300)
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
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, timeout=300)
        return {"path": out.stdout.strip()}
    except Exception as e:
        return {"path": "", "error": str(e)}


@app.route("/demo", methods=["POST"])
def demo():
    measure = request.form.get("measure", "reho")
    bold = _synthetic_bold()
    dim = bold.shape[0]
    off = -(dim / 2) * 3.0
    affine = np.array([[3, 0, 0, off], [0, 3, 0, off],
                       [0, 0, 3, off], [0, 0, 0, 1]], dtype=float)
    note = "Synthetic BOLD — demonstrates the layout, not real biology."
    if measure in ("alff", "falff"):
        alff, falff, mask = measures.alff_falff(bold, tr=2.0, band=(0.01, 0.08))
        m = alff if measure == "alff" else falff
        params = "TR 2.0s, band 0.01-0.08 Hz"
    elif measure == "rsfa":
        m, mask = measures.rsfa(bold)
        params = "temporal std of BOLD"
    elif measure == "reho":
        m, mask = measures.reho(bold, cluster=27)
        params = "cluster 27"
    else:
        return _page_error(f"unknown demo measure: {measure}")
    return _render("DEMO (synthetic)", measure, m, mask, params, note,
                   affine=affine, mode="demo", form=request.form)


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
