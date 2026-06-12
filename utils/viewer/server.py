"""
PosterForest Viewer — FastAPI server
-------------------------------------
Reads the outputs/ directory and exposes run metadata via REST API.
Serves index.html at / and output files at /outputs/.

Run from the project root:
    uvicorn utils.viewer.server:app --reload --port 8080
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime
import json, re, time

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VIEWER_DIR = Path(__file__).parent          # utils/viewer/
PROJECT_DIR = VIEWER_DIR.parent.parent      # project root
OUTPUTS_DIR = PROJECT_DIR / "outputs"

# ---------------------------------------------------------------------------
# Step definitions (folder prefix → human label)
# ---------------------------------------------------------------------------
STEPS = [
    ("01_parse_raw_poster",      "1. Parse Paper"),
    ("02_filter_images_tables",  "2. Filter & Refine"),
    ("03_generate_outline",      "3. Poster Outline"),
    ("04_generate_layout",       "4. Layout"),
    ("05_generate_content",      "5. Content"),
    ("06_step_06_modification_planning", "6. Feedback Loop"),
    ("07_generate_powerpoint",   "7. PowerPoint"),
    ("08_finalize_output",       "8. Final Output"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL_PAT = r"(?:vllm_qwen[0-9]*(?:_[0-9]+)*(?:_vl)?|4o(?:-mini)?)"
_RUN_RE = re.compile(rf"^({_MODEL_PAT})_({_MODEL_PAT})_(.+)$")

def _parse_run_name(name: str, run_path: Path | None = None) -> dict:
    """Parse 'YYYYMMDD_HHMMSS_model_t_model_v_paper_folder' into metadata."""
    parts = name.split("_", 2)
    try:
        dt = datetime.strptime(f"{parts[0]}_{parts[1]}", "%Y%m%d_%H%M%S")
        timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        timestamp = name[:15]

    rest = parts[2] if len(parts) > 2 else name
    m = _RUN_RE.match(rest)
    if m:
        model_t, model_v, paper_folder = m.group(1), m.group(2), m.group(3)
        models = f"{model_t}_{model_v}"
    else:
        tokens = rest.split("_")
        paper_folder = tokens[-1]
        models = "_".join(tokens[:-1])

    # 1st priority: paper_path stored in final_log.json (exact PDF used)
    pdf_stem, pdf_parent = _pdf_info_from_log(run_path) if run_path else (None, None)
    # 2nd priority: single PDF in the papers folder
    if pdf_stem is None:
        pdf_stem = _pdf_stem(paper_folder)
    paper_display = pdf_stem if pdf_stem else paper_folder
    # Resolve actual parent folder: from log > scan papers/ by stem > original
    if pdf_parent:
        paper_folder = pdf_parent
    elif paper_display:
        found = _find_paper_parent(paper_display)
        if found:
            paper_folder = found

    return {"timestamp": timestamp, "paper": paper_display, "paper_folder": paper_folder, "models": models}


def _pdf_info_from_log(run_path: Path) -> tuple[str | None, str | None]:
    """Read paper_path from final_log.json and return (pdf_stem, parent_folder)."""
    try:
        log_path = run_path / "08_finalize_output" / "final_log.json"
        if log_path.exists():
            data = json.loads(log_path.read_text())
            paper_path = data.get("paper_path")
            if paper_path:
                p = Path(paper_path)
                return p.stem, p.parent.name
    except Exception:
        pass
    return None, None


def _pdf_stem(paper_folder: str) -> str | None:
    """Return PDF stem only when exactly one PDF exists; avoids wrong name for multi-PDF folders."""
    papers_dir = PROJECT_DIR / "papers" / paper_folder
    if papers_dir.exists():
        pdfs = sorted(papers_dir.glob("*.pdf"))
        if len(pdfs) == 1:
            return pdfs[0].stem
    return None


def _find_paper_parent(paper_stem: str) -> str | None:
    """Scan papers/ recursively to find the parent folder of a PDF matching paper_stem."""
    papers_dir = PROJECT_DIR / "papers"
    if papers_dir.exists():
        for pdf in papers_dir.rglob("*.pdf"):
            if pdf.stem == paper_stem:
                return pdf.parent.name
    return None


def _step_status(run_path: Path) -> list[dict]:
    # Timing strategy:
    #   end_ts   = max file mtime inside the step directory (recursive).
    #              More accurate than directory mtime, which can lag when
    #              files are written to subdirectories, and avoids rounding
    #              artifacts that collapse fast steps to 0 s.
    #   start_ts = previous step's end_ts (sequential pipeline chaining).
    #   Step 0 start_ts = earliest file mtime (parse writes files progressively).
    #
    # All timestamps are kept as full-precision floats (no rounding) so that
    # sub-second durations (e.g. fast layout/outline steps) are representable.

    # First pass: collect file mtimes + end_ts for every step directory.
    raw = []
    for prefix, label in STEPS:
        d = run_path / prefix
        done = d.exists() and any(d.iterdir())
        end_ts = None
        file_mtimes: list[float] = []
        if done:
            try:
                file_mtimes = [f.stat().st_mtime for f in d.rglob("*") if f.is_file()]
                if file_mtimes:
                    end_ts = max(file_mtimes)
            except OSError:
                pass
        raw.append({
            "prefix": prefix, "label": label,
            "done": done, "end_ts": end_ts, "_mtimes": file_mtimes,
        })

    # Second pass: assign start_ts via chaining.
    statuses = []
    prev_end = None
    for i, item in enumerate(raw):
        if i == 0:
            # Step 1 (Parse) writes files progressively — use earliest mtime as start.
            start_ts = min(item["_mtimes"]) if item["_mtimes"] else None
        else:
            # All later steps: start when the previous step ended.
            start_ts = prev_end

        statuses.append({
            "prefix":   item["prefix"],
            "label":    item["label"],
            "done":     item["done"],
            "start_ts": start_ts,
            "end_ts":   item["end_ts"],
        })
        if item["end_ts"] is not None:
            prev_end = item["end_ts"]
    return statuses


def _is_stalled(run_path: Path, stall_minutes: int = 10) -> bool:
    """Return True if no activity in the run dir for stall_minutes.

    Newly created run dirs (within stall_minutes) are never considered stalled
    even if no output files exist yet (pipeline is still initializing).
    """
    cutoff = time.time() - stall_minutes * 60
    try:
        if run_path.stat().st_mtime > cutoff:
            return False
    except OSError:
        pass
    for p in run_path.rglob("*"):
        try:
            if p.stat().st_mtime > cutoff:
                return False
        except OSError:
            pass
    return True


def _figures(run_path: Path) -> list[str]:
    """Return web-relative paths to extracted paper figures."""
    fig_dir = run_path / "01_parse_raw_poster"
    paths = []
    for img in sorted(fig_dir.rglob("*.png")):
        if "figure" in img.name or "picture" in img.name or "table" in img.name:
            paths.append("/outputs/" + img.relative_to(OUTPUTS_DIR).as_posix())
    return paths


def _poster_sort_key(p: Path) -> tuple:
    """Chronological sort key for step6 attempt images.

    step6 saves end AFTER incrementing num_attempts, so attempt_01_end is
    the end of pass 0.  Correct order: start(pass N) → iters → end(pass N)
    which maps to attempt_N_start → attempt_N_iter_* → attempt_{N+1}_end.

    Sort key: (attempt_num, type_order, iter_num)
      end   → type 0  (comes before same-number start)
      start → type 1
      iter  → type 2+, with applied after non-applied within same iter
    """
    name = p.stem
    m = re.match(r"step6_attempt_(\d+)_(end|start|iter_(\d+)(.*)?)", name)
    if not m:
        return (999, 999, 999)
    attempt = int(m.group(1))
    kind = m.group(2)
    if kind == "end":
        return (attempt, 0, 0)
    if kind == "start":
        return (attempt, 1, 0)
    # iter_N[_suffix]: internal=0, applied=1, cropped skipped at glob level
    iter_num = int(m.group(3))
    suffix = m.group(4) or ""
    is_applied = 1 if "_applied" in suffix else 0
    return (attempt, 2, iter_num * 2 + is_applied)


def _intermediate_posters(run_path: Path) -> dict:
    """Return categorised step-6 images.

    Keys:
      full_posters – whole-poster snapshots (start / applied / end), non-internal
      panel_sims   – ALL per-panel simulation PNGs including _99 (final state)
    """
    tmp = run_path / "tmp"
    if not tmp.exists():
        return {"full_posters": [], "panel_sims": []}

    def url(p: Path) -> str:
        return "/outputs/" + p.relative_to(OUTPUTS_DIR).as_posix()

    full_posters = [
        url(p)
        for p in sorted(tmp.glob("step6_attempt*.jpg"), key=_poster_sort_key)
        if not p.stem.endswith("_cropped")  # exclude VLM-input crops (not meaningful poster views)
    ]

    # Include _99 files so JS can show before→after comparison per panel
    panel_sims = [url(p) for p in sorted(tmp.glob("step6_panel_*_simulation.png"))]

    return {"full_posters": full_posters, "panel_sims": panel_sims}


def _final_poster(run_path: Path) -> str | None:
    p = run_path / "08_finalize_output" / "poster_final.jpg"
    return "/outputs/" + p.relative_to(OUTPUTS_DIR).as_posix() if p.exists() else None


def _step_stats(run_path: Path) -> dict:
    """Collect per-step summary stats from output JSONs."""
    stats: dict = {}

    # Step 1: parse — count ALL assets (with + without captions)
    p = run_path / "01_parse_raw_poster"
    s1: dict = {}
    try:
        figs_cap    = len(json.loads((list(p.glob("*_images.json")) or [None])[0].read_text())) if list(p.glob("*_images.json")) else 0
        figs_nocap  = len(json.loads((list(p.glob("*_images_no_caption.json")) or [None])[0].read_text())) if list(p.glob("*_images_no_caption.json")) else 0
        tbls_cap    = len(json.loads((list(p.glob("*_tables.json")) or [None])[0].read_text())) if list(p.glob("*_tables.json")) else 0
        tbls_nocap  = len(json.loads((list(p.glob("*_tables_no_caption.json")) or [None])[0].read_text())) if list(p.glob("*_tables_no_caption.json")) else 0
        if figs_cap + figs_nocap:
            s1["figures"] = figs_cap + figs_nocap
        if tbls_cap + tbls_nocap:
            s1["tables"] = tbls_cap + tbls_nocap
    except Exception:
        pass
    for fname in p.glob("*_raw_content.json"):
        try:
            d = json.loads(fname.read_text())
            words = 0
            for sec in d.get("sections", []):
                words += len((sec.get("content") or "").split())
                for sub in sec.get("subsections", []):
                    words += len((sub.get("content") or "").split())
            s1["words"] = words
        except Exception:
            pass
    if s1:
        stats["step1"] = s1

    # Step 2: refined tree — sections + filtered asset counts
    p2 = run_path / "02_filter_images_tables"
    s2: dict = {}
    for fname in p2.glob("*_refined_tree.json"):
        try:
            d = json.loads(fname.read_text())
            meta = d.get("metadata", {})
            s2["sections"]    = meta.get("main_sections", 0)
            s2["subsections"] = meta.get("total_subsections", 0)
        except Exception:
            pass
    # Filtered asset counts come from step-3 figures.json (maps what survived filtering)
    p3 = run_path / "03_generate_outline"
    for fname in p3.glob("*_figures.json"):
        try:
            keys = list(json.loads(fname.read_text()).keys())
            s2["fig_kept"] = sum(1 for k in keys if "fig" in k and "table" not in k)
            s2["tbl_kept"] = sum(1 for k in keys if "table" in k)
        except Exception:
            pass
    if s2:
        stats["step2"] = s2

    # Step 3: panels
    for fname in p3.glob("*_poster_panels.json"):
        try:
            d = json.loads(fname.read_text())
            meta = d.get("metadata", {})
            stats["step3"] = {
                "panels":  meta.get("total_panels", 0),
                "assets":  meta.get("figures_mapped", 0),
            }
        except Exception:
            pass

    # Step 6: attempt count
    tmp = run_path / "tmp"
    if tmp.exists():
        attempts = len(list(tmp.glob("step6_attempt*_start.jpg")))
        if attempts:
            stats["step6"] = {"attempts": attempts}

    return stats


def _pipeline_data(run_path: Path) -> dict:
    """Step-by-step outline data. Always returns a dict (never None)."""
    result: dict = {}

    # ── Step 1: all extracted assets with full metadata
    p1 = run_path / "01_parse_raw_poster"
    assets: dict = {}  # id → {type, caption, url}  (used for step-2 cross-ref)
    step1_figs: list = []
    step1_tbls: list = []
    for f in sorted(p1.glob("*_images*.json")):
        try:
            for k, v in json.loads(f.read_text()).items():
                img_path = v.get("figure_path", "")
                entry = {
                    "id": str(k),
                    "type": "figure",
                    "caption": (v.get("caption") or "")[:140],
                    "url": "/" + img_path if img_path else None,
                }
                assets[str(k)] = entry
                step1_figs.append(entry)
        except Exception:
            pass
    for f in sorted(p1.glob("*_tables*.json")):
        try:
            for k, v in json.loads(f.read_text()).items():
                tbl_path = v.get("table_path") or v.get("figure_path") or v.get("path", "")
                entry = {
                    "id": f"t{k}",
                    "type": "table",
                    "caption": (v.get("caption") or "")[:140],
                    "url": "/" + tbl_path if tbl_path else None,
                }
                assets[f"t{k}"] = entry
                step1_tbls.append(entry)
        except Exception:
            pass
    if step1_figs or step1_tbls:
        result["step1"] = {"figures": step1_figs, "tables": step1_tbls}

    # ── Step 2: refined tree → section/subsection hierarchy
    p2 = run_path / "02_filter_images_tables"
    sections: list = []
    step2_kept_ids: set = set()
    step2_meta: dict = {}
    for f in p2.glob("*_refined_tree.json"):
        try:
            tree = json.loads(f.read_text())
            root = tree.get("tree_data", {})
            meta = tree.get("metadata", {})
            step2_meta = {
                "sections":    meta.get("main_sections", 0),
                "subsections": meta.get("total_subsections", 0),
            }

            def _parse(node: dict) -> dict:
                sec_assets = []
                for fig_id in node.get("assets", {}).get("figures", []):
                    key = str(fig_id)
                    if key in assets:
                        step2_kept_ids.add(key)
                        sec_assets.append(assets[key])
                for tbl_id in node.get("assets", {}).get("tables", []):
                    key = f"t{tbl_id}"
                    if key in assets:
                        step2_kept_ids.add(key)
                        sec_assets.append(assets[key])
                return {
                    "name":     node.get("section_name", ""),
                    "pid":      str(node.get("panel_id", "")),
                    "content":  (node.get("content") or "")[:500],
                    "assets":   sec_assets,
                    "children": [_parse(c) for c in node.get("children", [])],
                    "bullets":  [],
                }

            sections = [_parse(c) for c in root.get("children", [])]
            break
        except Exception:
            pass
    if sections:
        dropped = [
            {"id": k, "caption": v["caption"][:80], "type": v["type"], "url": v.get("url")}
            for k, v in assets.items() if k not in step2_kept_ids
        ]
        result["step2"] = {
            "sections": sections,
            "meta":     step2_meta,
            "kept_figs":   sum(1 for k in step2_kept_ids if assets.get(k, {}).get("type") == "figure"),
            "kept_tbls":   sum(1 for k in step2_kept_ids if assets.get(k, {}).get("type") == "table"),
            "dropped":     dropped[:20],
            "dropped_total": len(dropped),
        }

    # ── Step 3: poster outline (panels)
    p3 = run_path / "03_generate_outline"
    step3: dict = {}
    for fname in p3.glob("*_poster_panels.json"):
        try:
            d = json.loads(fname.read_text())
            meta = d.get("metadata", {})
            step3 = {
                "panels": meta.get("total_panels", 0),
                "assets": meta.get("figures_mapped", 0),
            }
        except Exception:
            pass
    if step3:
        result["step3"] = step3

    # ── Step 5: bullet contents → attach to step2 sections
    if sections:
        p5 = run_path / "05_generate_content"
        bullets_by_pid: dict = {}
        for f in p5.glob("*_bullet_contents.json"):
            try:
                bc = json.loads(f.read_text())
                if not isinstance(bc, list):
                    continue
                for item in bc:
                    pid  = str(item.get("panel_id", ""))
                    name = item.get("textbox_name", "")
                    if "title" in name.lower():
                        continue
                    texts = [
                        "".join(r.get("text", "") for r in para.get("runs", [])).strip()
                        for para in item.get("content_for_ppt", [])
                    ]
                    texts = [t for t in texts if t]
                    if texts:
                        bullets_by_pid.setdefault(pid, []).extend(texts)
            except Exception:
                pass

        def _attach(sec: dict) -> None:
            pid = sec["pid"]
            found = []
            for bpid, blist in bullets_by_pid.items():
                if bpid == pid or (pid and bpid.startswith(pid) and bpid != pid):
                    found.extend(blist)
            if found:
                sec["bullets"] = found
            for child in sec.get("children", []):
                _attach(child)

        for sec in sections:
            _attach(sec)

    # ── Keep legacy `sections` key for Activity tab pidLookup back-compat
    result["sections"] = sections

    return result


def _activity_log(run_path: Path) -> dict | None:
    """Read step6 activity log from tmp/step6_activity_log.json."""
    p = run_path / "tmp" / "step6_activity_log.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return None


def _token_usage(run_path: Path) -> dict | None:
    """Read token counts from 08_finalize_output/*step_data.json."""
    for step_data in sorted((run_path / "08_finalize_output").glob("*step_data.json")):
        try:
            data = json.loads(step_data.read_text())
            log = data.get("final_log.json", {})
            if "input_tokens_t" in log:
                return {
                    "input_t":    log.get("input_tokens_t", 0),
                    "output_t":   log.get("output_tokens_t", 0),
                    "input_v":    log.get("input_tokens_v", 0),
                    "output_v":   log.get("output_tokens_v", 0),
                    "time_taken": log.get("time_taken"),
                }
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="PosterForest Viewer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Serve output files directly
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


@app.get("/api/runs")
def list_runs():
    if not OUTPUTS_DIR.exists():
        return []
    runs = []
    for d in sorted(OUTPUTS_DIR.iterdir(), key=lambda p: p.name, reverse=True):
        if not d.is_dir() or not re.match(r"^\d{8}_\d{6}_", d.name):
            continue
        meta = _parse_run_name(d.name, run_path=d)
        steps = _step_status(d)
        completed = sum(1 for s in steps if s["done"])
        stalled = completed < len(STEPS) and _is_stalled(d)
        runs.append({
            "id":           d.name,
            "timestamp":    meta["timestamp"],
            "paper":        meta["paper"],
            "paper_folder": meta["paper_folder"],
            "models":       meta["models"],
            "completed": completed,
            "total":     len(STEPS),
            "stalled":   stalled,
        })
    return runs


@app.get("/api/runs/{run_id:path}")
def get_run(run_id: str):
    run_path = OUTPUTS_DIR / run_id
    if not run_path.exists():
        return JSONResponse(status_code=404, content={"error": "run not found"})

    steps = _step_status(run_path)
    completed = sum(1 for s in steps if s["done"])
    stalled = completed < len(STEPS) and _is_stalled(run_path)
    inter = _intermediate_posters(run_path)
    meta = _parse_run_name(run_id.split("/")[-1], run_path=run_path)
    return {
        "id":           run_id,
        "paper":        meta["paper"],
        "paper_folder": meta["paper_folder"],
        "models":       meta["models"],
        "steps":        steps,
        "completed":    completed,
        "total":        len(STEPS),
        "stalled":      stalled,
        "figures":      _figures(run_path),
        "intermediate": inter["full_posters"],   # full-poster snapshots
        "panel_sims":   inter["panel_sims"],     # per-panel overflow checks
        "final":        _final_poster(run_path),
        "tokens":       _token_usage(run_path),
        "stats":        _step_stats(run_path),
        "activity":     _activity_log(run_path),
        "pipeline":     _pipeline_data(run_path),
    }


@app.get("/")
def index():
    return FileResponse(VIEWER_DIR / "index.html")
