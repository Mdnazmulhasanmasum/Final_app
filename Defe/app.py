import io, json, time, hashlib
from html import escape as esc
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 40_000_000   # refuse decompression-bomb sized images
MAX_UPLOAD_MB = 15

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.graphics.shapes import Drawing, Rect
    from reportlab.platypus import (HRFlowable, Image as RLImage, KeepTogether, Paragraph,
                                    SimpleDocTemplate, Spacer, Table, TableStyle)
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

# ───────────────────────── PAGE ─────────────────────────
st.set_page_config(page_title="TinyBrainNet-MT", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{--bg:#0b1020;--surface:#111827;--line:#1f2a3d;--text:#e6edf3;--mute:#8b98a7;
      --accent:#5eead4;--warn:#fbbf24;--bad:#f87171;--ok:#4ade80;--r:12px}
html,body,[class*="css"],.stApp{font-family:'Inter',sans-serif}
.stApp{background:var(--bg);color:var(--text)}
header[data-testid="stHeader"]{background:transparent}
footer,#MainMenu{visibility:hidden}
.block-container{max-width:1180px;padding-top:2.2rem;padding-bottom:4rem}
section[data-testid="stSidebar"]{background:var(--surface);border-right:1px solid var(--line)}
h1.t{font-size:1.7rem;font-weight:700;letter-spacing:-.02em;margin:0}
p.s{color:var(--mute);font-size:.92rem;margin:.25rem 0 1.4rem}
h3{font-weight:600!important;letter-spacing:-.01em}

[data-testid="stFileUploaderDropzone"]{background:var(--surface);border:1.5px dashed #2f3b4a;
  border-radius:var(--r);padding:2.2rem 1rem;transition:border-color .2s,background .2s}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--accent);background:#142030}
[data-testid="stFileUploaderDropzone"] button{background:var(--accent);color:#04201c;border:0;
  border-radius:8px;font-weight:600;padding:.5rem 1.1rem}
[data-testid="stFileUploaderDropzone"] small,[data-testid="stFileUploaderDropzone"] span{color:var(--mute)}

.banner{border-radius:var(--r);padding:14px 18px;margin:1.1rem 0;border:1px solid var(--line);
  border-left:4px solid var(--c);background:var(--surface)}
.banner .tt{color:var(--c);font-weight:700}
.banner div{color:var(--mute);font-size:.9rem;margin-top:4px;line-height:1.55}
.banner div b{color:var(--text);font-weight:600}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);padding:14px 16px}
.kpi .l{color:var(--mute);font-size:.78rem}
.kpi .v{font-size:1.5rem;font-weight:700;margin-top:2px;letter-spacing:-.01em}
.cap{color:var(--mute);font-size:.8rem;text-align:center;margin-bottom:6px}
.bar{margin:0 0 12px}
.bar .row{display:flex;justify-content:space-between;font-size:.88rem;margin-bottom:5px}
.bar .row span:last-child{color:var(--mute);font-variant-numeric:tabular-nums}
.bar .track{height:6px;background:var(--line);border-radius:99px;overflow:hidden}
.bar .fill{height:100%;border-radius:99px;background:#3b4756}
.bar.top .fill{background:var(--accent)}
.chk{display:flex;gap:10px;align-items:flex-start;font-size:.88rem;margin-bottom:9px;line-height:1.45}
.chk i{width:8px;height:8px;border-radius:50%;margin-top:6px;flex:none}
.note{color:var(--mute);font-size:.78rem;margin-top:2.2rem;line-height:1.6}
</style>
""", unsafe_allow_html=True)

# ───────────────────────── MODEL ─────────────────────────
BASE = Path(__file__).resolve().parent
MODEL_PATHS = [BASE / "model" / "tinybrainnet.onnx", BASE / "tinybrainnet.onnx",
               BASE / "model" / "uninet.onnx", BASE.parent / "model" / "tinybrainnet.onnx"]
META_PATHS = [BASE / "model" / "meta.json", BASE / "meta.json", BASE.parent / "model" / "meta.json"]
SAMPLES_DIR = BASE / "samples" if (BASE / "samples").exists() else BASE.parent / "samples"
MODEL_PATH = next((p for p in MODEL_PATHS if p.exists()), None)
META_PATH = next((p for p in META_PATHS if p.exists()), None)

if MODEL_PATH is None:
    st.error("Model file not found. Put `tinybrainnet.onnx` inside a `model/` folder next to app.py.")
    st.stop()


def read_meta(p):
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p else {}
    except Exception:
        return {}


@st.cache_resource(show_spinner="Loading model…")
def load_session(path):
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    o.inter_op_num_threads = 1
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=o, providers=["CPUExecutionProvider"])


meta = read_meta(META_PATH)
sess = load_session(str(MODEL_PATH))

CLASSES = [str(c) for c in meta.get("classes", ["Glioma", "Meningioma", "Pituitary", "No_tumor"])]
NO_TUMOR = next((i for i, c in enumerate(CLASSES) if c.lower() == "no_tumor"), None)
TUMOR_IDX = [i for i in range(len(CLASSES)) if i != NO_TUMOR]
INP = meta.get("input", {})
IMG = int(INP.get("size", 224))
MEAN = np.array(INP.get("mean", [0.1838] * 3), np.float32)
STD = np.array(INP.get("std", [0.1984] * 3), np.float32)
TEMP = max(float(meta.get("calibration", {}).get("temperature", 0.693)), 1e-4)
SEG_IS_PROB = bool(meta.get("outputs", {}).get("segmentation_is_probability", False))
IN_NAME = sess.get_inputs()[0].name
OUT_NAMES = [o.name for o in sess.get_outputs()][:3]
APP_DEF = meta.get("app_defaults", {})   # written by notebook Part 17 (picked from data)
MIN_PX = max(int(meta.get("segmentation", {}).get("min_area_px", 20)), 20)
AUTO_T = float(meta.get("triage", {}).get("auto", 0.88))      # >= auto  -> confident
REVIEW_T = float(meta.get("triage", {}).get("review", 0.51))  # < review -> uncertain

# ---- decision rules. Every number here can be overridden from meta.json -> "fusion"
#      (write them from the notebook together with how they were picked, so the thesis can cite it).
FUS = meta.get("fusion", {})
DEC_THR = float(meta.get("segmentation", {}).get("threshold", 0.50))  # FIXED mask threshold that drives the diagnosis
MIN_LESION_PCT = float(FUS.get("min_lesion_pct", 0.40))   # smaller mask area than this = "no lesion"
COMP_MEAN = float(FUS.get("comp_mean", 0.60))             # a mask fragment must have this mean confidence ...
COMP_PEAK = float(FUS.get("comp_peak", 0.80))             # ... and this peak confidence to survive clean-up
STRONG_CONF = float(FUS.get("strong_lesion_conf", 0.85))  # "strong lesion": mean mask probability
STRONG_CONS = float(FUS.get("strong_lesion_views", 0.60)) # ... seen in this fraction of views
STRONG_AREA = float(FUS.get("strong_lesion_area", 1.0))   # ... and covering at least this % of the slice
CONFLICT_PTUMOUR = float(FUS.get("conflict_p_tumour", 0.35))  # No_tumor head vs strong lesion: tumour prob >= this -> conflict
CONFLICT_PCLEAR = float(FUS.get("conflict_p_clear", 0.60))    # No_tumor head vs any lesion: P(No_tumor) < this -> conflict

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("Asia/Dhaka")
except Exception:                       # tzdata missing (e.g. some Windows installs)
    TZ = timezone(timedelta(hours=6))   # Bangladesh time

# ───────────────────────── PIPELINE ─────────────────────────
def auto_crop(pil):
    g = np.array(pil.convert("L"))
    _, th = cv2.threshold(g, 22, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > g.size * 0.10:
            x, y, w, h = cv2.boundingRect(c)
            p = 6
            return pil.crop((max(0, x - p), max(0, y - p), min(g.shape[1], x + w + p), min(g.shape[0], y + h + p)))
    return pil


def base_rgb(pil, crop):
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    if crop:
        pil = auto_crop(pil)
    return cv2.resize(np.asarray(pil, np.uint8), (IMG, IMG), interpolation=cv2.INTER_AREA)


def to_tensor(rgb, robust):
    x = rgb.astype(np.float32) / 255.0
    if robust:
        tissue = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) > 20
        if tissue.sum() > 100:
            lo, hi = np.percentile(x[tissue], [1, 99])
            x = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    return ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


def v_clahe(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def v_clean(rgb):
    """Median filter wipes thin watermark text / scratches but keeps the large structures."""
    return cv2.medianBlur(rgb, 5)


def v_gamma(rgb, g):
    return (np.power(rgb.astype(np.float32) / 255.0, g) * 255).astype(np.uint8)


def v_rot(rgb, deg):
    M = cv2.getRotationMatrix2D((IMG / 2, IMG / 2), deg, 1.0)
    return cv2.warpAffine(rgb, M, (IMG, IMG), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))


def v_zoom(rgb, f):
    if f > 1:
        m = int(IMG * (1 - 1 / f) / 2)
        return cv2.resize(rgb[m:IMG - m, m:IMG - m], (IMG, IMG), interpolation=cv2.INTER_CUBIC)
    s = int(IMG * f)
    out = np.zeros_like(rgb)
    o = (IMG - s) // 2
    out[o:o + s, o:o + s] = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_AREA)
    return out


def brain_mask(rgb):
    """Soft mask of the head region, used to keep the attribution map off the background."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    m = (g > 20).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))  # stay inside the skull ring
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), 4)


def head_region(rgb):
    """Filled head silhouette (not eroded, so tumours touching the skull are kept). Removes background/watermark blobs."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    m = (g > 20).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.ones(g.shape, bool)
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < g.size * 0.10:
        return np.ones(g.shape, bool)
    out = np.zeros(g.shape, np.uint8)
    cv2.drawContours(out, [c], -1, 1, -1)
    return out.astype(bool)


def ood_flags(img_bytes, disp):
    """Cheap checks that the upload looks like a grey-scale brain MRI slice. Returns a list of warnings.
    It cannot prove the image is a brain MRI, it only catches obvious non-MRI input (colour photos, full-frame images)."""
    flags = []
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
        im.thumbnail((256, 256))
        a = np.asarray(im, np.int16)
        chroma = float(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean())
        if chroma > 12:
            flags.append("The image is coloured, but MRI slices are normally grey-scale.")
    except Exception:
        pass
    g = cv2.cvtColor(disp, cv2.COLOR_RGB2GRAY)
    fg = float((g > 20).mean())
    if fg < 0.15 or fg > 0.97:
        flags.append("No clear head silhouette on a dark background was found.")
    k = max(IMG // 10, 4)
    corners = np.concatenate([g[:k, :k].ravel(), g[:k, -k:].ravel(), g[-k:, :k].ravel(), g[-k:, -k:].ravel()])
    if float(corners.mean()) > 60:
        flags.append("The image corners are bright, while a brain slice normally has a dark background.")
    return flags


def stretch(fn, *a, **k):
    """Full-width widget on both new (width='stretch') and older (use_container_width) Streamlit versions."""
    try:
        return fn(*a, width="stretch", **k)
    except Exception:
        return fn(*a, use_container_width=True, **k)


def clean_mask(sp, thr, region):
    """Threshold, keep only the head region, then drop fragments that are tiny or weakly confident.
    This is what removes most false-positive masks on healthy slices."""
    m = (sp >= thr) & region
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    out = np.zeros(m.shape, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_PX:
            continue
        comp = lab == i
        if sp[comp].mean() >= COMP_MEAN and sp[comp].max() >= COMP_PEAK:
            out |= comp
    return out


def run_once(t):
    res = sess.run(OUT_NAMES, {IN_NAME: t})
    cls = np.squeeze(res[0]).astype(np.float64)
    seg = np.squeeze(res[1]).astype(np.float32) if len(res) > 1 else None
    attr = np.squeeze(res[2]).astype(np.float32) if len(res) > 2 else None
    return cls, seg, attr


@st.cache_data(show_spinner="Analysing scan…", max_entries=6, ttl=900)
def infer(img_bytes, crop, robust, multiview):
    """Cached, so sliders and toggles that don't change the network input never re-run it.
    Multi-view = 10 views: original, flip, CLAHE, watermark-clean, 2 gammas, 2 zooms, 2 rotations.
    Segmentation is averaged over every view that keeps the geometry (orig, flip, CLAHE, clean, gammas).
    Saliency is averaged over original+flip only."""
    rgb = base_rgb(Image.open(io.BytesIO(img_bytes)), crop)
    t0 = time.perf_counter()
    c0, s0, a0 = run_once(to_tensor(rgb, robust))
    logits = [c0]
    segs = [s0] if s0 is not None else []
    attr = a0
    if multiview:
        c1, s1, a1 = run_once(to_tensor(np.ascontiguousarray(rgb[:, ::-1]), robust))
        logits.append(c1)
        if s1 is not None:
            segs.append(s1[..., ::-1])
        if a0 is not None and a1 is not None:
            attr = (a0 + a1[..., ::-1]) / 2
        for v in (v_clahe(rgb), v_clean(rgb), v_gamma(rgb, 0.8), v_gamma(rgb, 1.25)):
            c, s, _ = run_once(to_tensor(v, robust))
            logits.append(c)
            if s is not None:
                segs.append(s)
        for v in (v_zoom(rgb, 1.12), v_zoom(rgb, 0.88), v_rot(rgb, 7), v_rot(rgb, -7)):
            logits.append(run_once(to_tensor(v, robust))[0])
    seg = np.mean(segs, axis=0) if segs else None
    seg_views = np.stack(segs) if segs else None
    return rgb, np.stack(logits), seg, seg_views, attr, (time.perf_counter() - t0) * 1000


def softmax(z, temp=1.0):
    z = np.asarray(z, np.float64).ravel() / temp
    e = np.exp(np.clip(z - z.max(), -60, 0))
    return e / e.sum()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def blend_mask(img, mask, alpha):
    out = img.copy()
    out[mask] = (out[mask] * (1 - alpha) + np.array([239, 68, 68]) * alpha).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, (255, 80, 80), 1)
    return out


def blend_heat(img, attr, alpha):
    heat = cv2.cvtColor(cv2.applyColorMap((attr * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    w = (attr[..., None] * alpha)  # fade out where attribution is ~0 so the scan stays readable
    return (img * (1 - w) + heat * w).astype(np.uint8)


def titled(img, text):
    bar = np.full((30, img.shape[1], 3), (11, 16, 32), np.uint8)
    cv2.putText(bar, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 237, 243), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def png_bytes(arr, size=None):
    if size:
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_CUBIC)
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="PNG")
    return b.getvalue()


# ───────────────────────── PDF REPORT ─────────────────────────
def latin(s):
    return esc((s or "").encode("latin-1", "replace").decode("latin-1"))


def build_pdf(info, r, panels, captions):
    buf = io.BytesIO()
    L = 16 * mm
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=L, rightMargin=L, topMargin=14 * mm, bottomMargin=20 * mm,
                            title="Brain MRI AI Analysis Report", author="TinyBrainNet-MT")
    W = A4[0] - 2 * L
    ink, mute, line, acc = (colors.HexColor(c) for c in ("#0f172a", "#64748b", "#e2e8f0", "#0d9488"))
    ss = getSampleStyleSheet()
    S = lambda n, **k: ParagraphStyle(n, parent=ss["Normal"], **{"textColor": ink, **k})
    h1 = S("h1", fontName="Helvetica-Bold", fontSize=18, leading=22)
    sub = S("sub", fontSize=9, leading=12, textColor=mute)
    h2 = S("h2", fontName="Helvetica-Bold", fontSize=11, leading=14, spaceBefore=12, spaceAfter=5)
    body = S("body", fontSize=9.5, leading=14)
    lab = S("lab", fontSize=7.5, leading=10, textColor=mute)
    val = S("val", fontName="Helvetica-Bold", fontSize=10, leading=13)
    big = S("big", fontName="Helvetica-Bold", fontSize=15, leading=18)
    cap = S("cap", fontSize=8, leading=10, textColor=mute, alignment=1)
    P = Paragraph

    def bar(p, w=70 * mm, h=3.2 * mm, c=acc):
        d = Drawing(w, h)
        d.add(Rect(0, 0, w, h, fillColor=line, strokeColor=None, rx=1.6, ry=1.6))
        d.add(Rect(0, 0, max(w * p, 0.5), h, fillColor=c, strokeColor=None, rx=1.6, ry=1.6))
        return d

    story = [P("Brain MRI AI Analysis Report", h1),
             P(f"RESEARCH PROTOTYPE &middot; TinyBrainNet-MT &middot; Khwaja Yunus Ali University &middot; Report ID {r['rid']}", sub),
             Spacer(1, 3), HRFlowable(width="100%", thickness=1.2, color=acc, spaceAfter=8)]

    cw = [27 * mm, W / 2 - 27 * mm] * 2
    rows = [[P("Patient name", lab), P(latin(info["patient"]) or "Not provided", val), P("Patient ID", lab), P(latin(info["pid"]) or "Not provided", val)],
            [P("Age / Sex", lab), P(latin(info["agesex"]) or "Not provided", val), P("Report date", lab), P(info["date"], val)],
            [P("Doctor", lab), P(latin(info["doctor"]) or "Not provided", val), P("Designation", lab), P(latin(info["desig"]) or "Not provided", val)],
            [P("Facility", lab), P(latin(info["facility"]) or "Not provided", val), P("Scan file", lab), P(r["sha"], val)]]
    t = Table(rows, colWidths=cw)
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, line), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LEFTPADDING", (0, 0), (-1, -1), 2)]))
    story += [t, P("AI findings", h2)]

    lv = {"High": "#16a34a", "Medium": "#d97706", "Low": "#dc2626"}[r["level"]]
    res = Table([[P("Result", lab), P(r["conf_label"], lab), P("Lesion area", lab), P("Reliability", lab)],
                 [P(latin(r["name"]), big), P(f'{r["conf"]*100:.1f}%', big), P(f'{r["area"]:.2f}%', big),
                  P(f'<font color="{lv}">{r["level"]}</font>', big)]],
                colWidths=[W * .34, W * .22, W * .22, W * .22])
    res.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")), ("BOX", (0, 0), (-1, -1), 0.6, line),
                             ("LEFTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story += [res, Spacer(1, 6), P(f'<b>Impression:</b> {r["title"]}', body), Spacer(1, 2), P(r["summary"], body),
              P("Visual inspection", h2)]

    pw = (W - 8 * mm) / len(panels)
    imgs = Table([[RLImage(io.BytesIO(p), pw, pw) for p in panels], [P(c, cap) for c in captions]], colWidths=[pw] * len(panels))
    imgs.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    story.append(imgs)

    story.append(P("Class probabilities", h2))
    prow = [[P(latin(CLASSES[i]), body), P(f'{r["probs"][i]*100:.1f}%', body),
             bar(float(r["probs"][i]), c=acc if i == r["pred"] else colors.HexColor("#94a3b8"))] for i in np.argsort(-r["probs"])]
    pt = Table(prow, colWidths=[40 * mm, 22 * mm, W - 62 * mm])
    pt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    story.append(pt)

    if r["types"]:
        story.append(P("Tumour type ranking (tumour classes only)", h2))
        trow = [[P(latin(n), body), P(f'{p*100:.1f}%', body), bar(float(p), c=colors.HexColor("#0d9488") if k == 0 else colors.HexColor("#94a3b8"))]
                for k, (n, p) in enumerate(r["types"])]
        tt = Table(trow, colWidths=[40 * mm, 22 * mm, W - 62 * mm])
        tt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        story.append(tt)

    story.append(P("Reliability checks", h2))
    for ok, text in r["checks"]:
        story.append(P(f'<font color="{"#16a34a" if ok else "#d97706"}"><b>{"OK" if ok else "CHECK"}</b></font> &nbsp;{text}', body))

    if info["notes"].strip():
        story += [P("Clinical notes", h2), P(latin(info["notes"]).replace("\n", "<br/>"), body)]

    sign = Table([[P("Reporting doctor", lab), "", P("Date", lab)],
                  [P(latin(info["doctor"]) or " ", val), "", P(info["date"], val)],
                  [P(latin(info["desig"]) or " ", sub), "", P("Signature / Seal", lab)]],
                 colWidths=[W * .5, W * .1, W * .4], rowHeights=[12, 40, 14])
    sign.setStyle(TableStyle([("LINEBELOW", (0, 1), (0, 1), 0.6, ink), ("LINEBELOW", (2, 1), (2, 1), 0.6, ink), ("VALIGN", (0, 1), (-1, 1), "BOTTOM")]))
    story.append(KeepTogether([Spacer(1, 16), sign]))

    story += [Spacer(1, 10), P("Settings: " + latin(r["settings"]), sub)]

    def footer(c, d):
        c.saveState()                      # diagonal watermark (drawn first, so it sits behind the content)
        c.setFillColor(colors.Color(0.5, 0.5, 0.5, alpha=0.10))
        c.setFont("Helvetica-Bold", 54)
        c.translate(A4[0] / 2, A4[1] / 2)
        c.rotate(35)
        c.drawCentredString(0, 0, "RESEARCH PROTOTYPE")
        c.restoreState()
        c.saveState()
        c.setStrokeColor(line)
        c.line(L, 14 * mm, A4[0] - L, 14 * mm)
        c.setFont("Helvetica", 7.5)
        c.setFillColor(mute)
        c.drawString(L, 9.5 * mm, "TinyBrainNet-MT research prototype. Not a medical device. A qualified clinician must review every finding.")
        c.drawRightString(A4[0] - L, 9.5 * mm, f"Page {d.page}")
        c.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


# ───────────────────────── SIDEBAR ─────────────────────────
with st.sidebar:
    st.markdown("#### Settings")
    samples = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png") for p in SAMPLES_DIR.glob(ext)) if SAMPLES_DIR.exists() else []
    sample_choice = st.selectbox("Try a sample", ["None"] + [p.name for p in samples])
    st.divider()
    crop = st.toggle("Auto-crop brain", True)
    robust = st.toggle("Intensity alignment", bool(APP_DEF.get("intensity_alignment", False)),
                       help="Percentile contrast stretch. Off by default: the original model was trained without it.")
    multiview = st.toggle("Multi-view check", True,
                          help="Runs the scan 10 ways (original, flip, contrast, watermark-clean, 2 gammas, 2 zooms, "
                               "2 rotations) and averages. Steadier on photos and web images that look unlike the training data.")
    soften = st.toggle("Soften probabilities", True,
                       help=f"On: temperature is at least 1.0, so probabilities are never sharper than the raw softmax "
                            f"(stops 100% on unfamiliar scans). Off: use the calibrated temperature from meta.json (T = {TEMP:.2f}).")
    seg_thr = st.slider("Overlay mask threshold (display only)", 0.10, 0.90,
                        float(np.clip(round(DEC_THR / 0.05) * 0.05, 0.10, 0.90)), 0.05,
                        help=f"Only changes the red overlay. The diagnosis always uses the fixed threshold {DEC_THR:.2f}.")
    alpha = st.slider("Overlay strength", 0.1, 0.9, 0.50, 0.05)
    st.divider()
    if st.button("Clear cached scans", help="Removes analysed scans from the server's memory."):
        st.cache_data.clear()
        st.toast("Cache cleared")
    st.caption("Privacy: scans are kept in memory for at most 15 minutes. Do not upload identifiable patient scans to a public deployment.")
    with st.expander("Report defaults (kept for every scan)"):
        def_doctor = st.text_input("Doctor name", key="def_doctor", placeholder="Dr. ...")
        def_desig = st.text_input("Designation / Reg. no.", key="def_desig")
        def_facility = st.text_input("Hospital / Clinic", key="def_facility")
    st.caption("Khwaja Yunus Ali University · CSE thesis prototype")

# ───────────────────────── HEADER + UPLOAD ─────────────────────────
st.markdown('<h1 class="t">🧠 TinyBrainNet-MT</h1>'
            '<p class="s">Upload a brain MRI slice to classify the tumour type and outline the lesion.</p>',
            unsafe_allow_html=True)
upload = st.file_uploader("Upload MRI", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

img_bytes = None
if upload is not None:
    img_bytes = upload.getvalue()
elif sample_choice != "None":
    img_bytes = (SAMPLES_DIR / sample_choice).read_bytes()
if img_bytes is None:
    st.caption("Drag and drop a scan above, or pick a sample from the sidebar.")
    st.stop()
if len(img_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
    st.error(f"This file is larger than {MAX_UPLOAD_MB} MB. Please upload a smaller image.")
    st.stop()

# ───────────────────────── INFERENCE ─────────────────────────
try:
    disp, logits, seg_raw, seg_views, attr_raw, ms = infer(img_bytes, crop, robust, multiview)
except Exception as e:
    st.error(f"Could not analyse this image: {e}")
    st.stop()

T = max(TEMP, 1.0) if soften else TEMP
mean_logits = logits.mean(0)
probs = softmax(mean_logits, T)
pred = int(probs.argmax())
conf = float(probs[pred])
n_views = len(logits)
agree = float((logits.argmax(1) == pred).mean())
p_clear = float(probs[NO_TUMOR]) if NO_TUMOR is not None else 0.0
p_tumour = 1.0 - p_clear
pred_is_clear = NO_TUMOR is not None and pred == NO_TUMOR

# ---- segmentation: clean the mask, then measure how consistent it is across views
region = head_region(disp)
if seg_raw is not None:
    sig_ = (lambda x: x) if SEG_IS_PROB else sigmoid
    sp = cv2.resize(sig_(seg_raw).astype(np.float32), (IMG, IMG), interpolation=cv2.INTER_LINEAR)
    raw_area = float((sp >= DEC_THR).mean()) * 100
    mask = clean_mask(sp, DEC_THR, region)
    view_has = []
    for sv in seg_views:
        spv = cv2.resize(sig_(sv).astype(np.float32), (IMG, IMG), interpolation=cv2.INTER_LINEAR)
        view_has.append(clean_mask(spv, DEC_THR, region).mean() * 100 >= MIN_LESION_PCT)
    seg_cons = float(np.mean(view_has))
else:
    sp = np.zeros((IMG, IMG), np.float32)
    mask = np.zeros((IMG, IMG), bool)
    raw_area, seg_cons = 0.0, 0.0
area = float(mask.mean()) * 100
les_conf = float(sp[mask].mean()) if mask.any() else 0.0
has_lesion = area >= MIN_LESION_PCT
strong_lesion = has_lesion and les_conf >= STRONG_CONF and seg_cons >= STRONG_CONS and area >= STRONG_AREA

# ---- fuse the two heads into ONE decision (this replaces the old "two heads, two answers" behaviour)
#   clear     : classifier No_tumor, no lesion
#   fp        : classifier clearly No_tumor, mask is weak -> mask treated as false positive and hidden
#   conflict  : classifier No_tumor but the lesion is strong / classifier not sure -> inconclusive, manual review
#   tumour    : classifier tumour + lesion
#   no_mask   : classifier tumour, no lesion outlined
if pred_is_clear and not has_lesion:
    state = "clear"
elif pred_is_clear and has_lesion:
    state = "conflict" if (strong_lesion and p_tumour >= CONFLICT_PTUMOUR) or p_clear < CONFLICT_PCLEAR else "fp"
elif has_lesion:
    state = "tumour"
else:
    state = "no_mask"

if state == "fp":                       # hide the suspicious mask so the report is self-consistent
    mask = np.zeros((IMG, IMG), bool)
    area, les_conf = 0.0, 0.0
conflict = state == "conflict"

# Overlay follows the display slider; the diagnosis above never does.
if seg_raw is not None and state in ("tumour", "conflict", "no_mask"):
    mask_view = clean_mask(sp, seg_thr, region)
else:
    mask_view = np.zeros((IMG, IMG), bool)

# Tumour type, ranked among tumour classes only (softened harder when the heads disagree)
T_type = max(T, 2.0) if conflict else T
tp = softmax(mean_logits[TUMOR_IDX], T_type)
order = np.argsort(-tp)
n1, p1 = CLASSES[TUMOR_IDX[order[0]]], float(tp[order[0]])
n2, p2 = (CLASSES[TUMOR_IDX[order[1]]], float(tp[order[1]])) if len(order) > 1 else ("", 0.0)
type_rank = [(CLASSES[TUMOR_IDX[j]], float(tp[j])) for j in order]

conf_label = "Confidence"
if state in ("clear", "fp"):
    final_idx, final_name, final_conf = NO_TUMOR, "No_tumor", p_clear
elif state == "tumour":
    final_idx, final_conf = pred, conf
    final_name = CLASSES[pred] if conf >= REVIEW_T else "Tumour (type uncertain)"
elif state == "no_mask":
    final_idx, final_conf = pred, conf
    final_name = CLASSES[pred] if (conf >= AUTO_T and agree >= 0.8) else "Inconclusive"
else:  # conflict
    final_idx, final_name, final_conf = TUMOR_IDX[order[0]], "Inconclusive", p_tumour
    conf_label = "Tumour probability"
if state == "no_mask" and final_name == "Inconclusive":
    conf_label = "Top-class prob."

# ---- reliability
if state in ("conflict",) or (state == "no_mask" and final_name == "Inconclusive"):
    level = "Low"
else:
    score = 0.5 * final_conf + 0.5 * agree
    level = "High" if (score >= 0.85 and final_conf >= AUTO_T) else "Medium" if score >= 0.60 else "Low"
    if state in ("fp", "no_mask") and level == "High":
        level = "Medium"
    if state == "tumour" and (les_conf < 0.70 or seg_cons < 0.5) and level == "High":
        level = "Medium"
    if final_conf < REVIEW_T:
        level = "Low"

ood = ood_flags(img_bytes, disp)
if ood:   # cap reliability when the input does not look like a brain MRI
    rank = {"Low": 0, "Medium": 1, "High": 2}
    cap_level = "Low" if len(ood) >= 2 else "Medium"
    if rank[level] > rank[cap_level]:
        level = cap_level

checks = [(agree >= 0.8, f"{int(round(agree * n_views))} of {n_views} views give the same class.")]
if state == "conflict":
    checks.append((False, f"Classifier says No_tumor ({p_clear*100:.0f}%) but segmentation outlined {area:.2f}% of the slice. Result held as inconclusive."))
elif state == "fp":
    checks.append((False, f"Weak mask ({raw_area:.2f}% before clean-up) on a scan the classifier calls No_tumor ({p_clear*100:.0f}%): treated as a false positive and hidden."))
elif state == "no_mask":
    checks.append((False, f"Classifier says {CLASSES[pred]} but segmentation found no clear lesion."))
else:
    checks.append((True, "Classifier and segmentation agree."))
if raw_area > area + 0.05 and state != "fp":
    checks.append((True, f"Clean-up removed {raw_area - area:.2f}% of weak or tiny mask fragments."))
if has_lesion and state != "fp":
    checks.append((les_conf >= 0.70, f"Lesion confidence (mean mask probability) {les_conf*100:.0f}%; lesion seen in {int(round(seg_cons*len(seg_views)))} of {len(seg_views)} views."))
if final_conf >= AUTO_T:
    checks.append((True, f"Confidence {final_conf*100:.0f}% is above the auto threshold ({AUTO_T*100:.0f}%)."))
elif final_conf >= REVIEW_T:
    checks.append((False, f"Confidence {final_conf*100:.0f}% is in the review band ({REVIEW_T*100:.0f}-{AUTO_T*100:.0f}%): a clinician should confirm."))
else:
    checks.append((False, f"Confidence {final_conf*100:.0f}% is below {REVIEW_T*100:.0f}%: the model is not sure."))
if not multiview:
    checks.append((False, "Multi-view check is off. Turn it on for a steadier result."))
for f_ in ood:
    checks.append((False, "Input check: " + f_))
checks.append((True, f"Analysed in {ms:.0f} ms on CPU."))

# Saliency for the reported class, kept inside the head region
attr = None
if attr_raw is not None and attr_raw.ndim == 3:
    a = cv2.resize(np.maximum(attr_raw[final_idx if final_idx < attr_raw.shape[0] else 0], 0), (IMG, IMG), interpolation=cv2.INTER_CUBIC)
    a = a * brain_mask(disp)
    lo, hi = np.percentile(a, [1, 99.5])
    if hi > lo + 1e-7:
        attr = np.clip((a - lo) / (hi - lo), 0, 1)

# ───────────────────────── RESULT ─────────────────────────
if state == "conflict":
    title = "Inconclusive: classifier and segmentation disagree"
    summary = (f"The classifier says <b>No_tumor</b> ({p_clear*100:.0f}%) but the segmentation head outlined a mass covering "
               f"<b>{area:.2f}%</b> of the slice. No final diagnosis is given. If a tumour is present the closest type is "
               f"<b>{n1}</b> ({p1*100:.0f}%), then {n2} ({p2*100:.0f}%). Please review manually.")
elif state == "fp":
    title = "No tumour detected (weak mask hidden)"
    summary = (f"<b>No_tumor</b> at {p_clear*100:.1f}%. A faint mask ({raw_area:.2f}% before clean-up) was removed as a probable "
               f"false positive. Reliability is capped at Medium.")
elif state == "no_mask":
    if final_name == "Inconclusive":
        title = "Inconclusive: tumour suspected, no lesion outlined"
        summary = (f"The classifier leans to <b>{CLASSES[pred]}</b> ({conf*100:.1f}%) but is below the auto threshold and the segmentation "
                   f"head found no lesion at threshold {DEC_THR:.2f}. Check the scan manually.")
    else:
        title = f"{final_name} predicted, no clear lesion outlined"
        summary = (f"The classifier is confident about <b>{final_name}</b> ({conf*100:.1f}%), but the segmentation head did not outline a "
                   f"lesion above the {DEC_THR:.2f} threshold. Check the scan manually.")
elif state == "clear":
    title = "No tumour detected"
    summary = f"Both heads agree. <b>No_tumor</b> at {p_clear*100:.1f}% confidence, no lesion outlined."
else:
    if final_name == "Tumour (type uncertain)":
        title = "Tumour found, type uncertain"
        summary = (f"Both heads see an abnormality (lesion {area:.2f}% of the slice) but the type confidence is only {conf*100:.0f}%. "
                   f"Closest types: <b>{n1}</b> ({p1*100:.0f}%), then {n2} ({p2*100:.0f}%). Please review manually.")
    else:
        title = f"{final_name} detected"
        summary = (f"Both heads agree. <b>{final_name}</b> at {conf*100:.1f}% confidence, lesion covers <b>{area:.2f}%</b> of the slice."
                   + ("" if level == "High" else " Reliability is not high, so review before relying on it."))

tone = "var(--ok)" if level == "High" else "var(--bad)" if level == "Low" else "var(--warn)"
st.markdown(f'<div class="banner" style="--c:{tone}"><span class="tt">{title}</span><div>{summary}</div></div>', unsafe_allow_html=True)
if ood:
    st.warning("This image may not be a brain MRI slice. " + " ".join(ood) + " Treat the result as unreliable.")

lvl_col = {"High": "var(--ok)", "Medium": "var(--warn)", "Low": "var(--bad)"}[level]
k = st.columns(4)
for col, lab_, v_, sty in [(k[0], "Result", final_name, ""),
                           (k[1], conf_label, f"{final_conf*100:.1f}%", ""),
                           (k[2], "Lesion area", f"{area:.2f}%", ""),
                           (k[3], "Reliability", level, f"color:{lvl_col}")]:
    col.markdown(f'<div class="kpi"><div class="l">{lab_}</div><div class="v" style="{sty}">{v_}</div></div>', unsafe_allow_html=True)

# ───────────────────────── VISUAL INSPECTION ─────────────────────────
st.markdown("### Visual inspection")
overlay = blend_mask(disp, mask_view, alpha) if mask_view.any() else disp
heat = blend_heat(disp, attr, alpha) if attr is not None else None
panels = [disp, overlay] + ([heat] if heat is not None else [])
captions = ["Resized model input", f"Predicted mask · overlay threshold {seg_thr:.2f}"] + (["Model attribution (experimental)"] if heat is not None else [])
for col, img_, cap_ in zip(st.columns(len(panels), gap="medium"), panels, captions):
    col.markdown(f'<div class="cap">{cap_}</div>', unsafe_allow_html=True)
    stretch(col.image, img_)

st.write("")
cl, cr = st.columns([3, 2], gap="large")
with cl:
    st.markdown("### Class probabilities")
    html_ = ""
    for i in np.argsort(-probs):
        html_ += (f'<div class="bar{" top" if i == pred else ""}"><div class="row"><span>{CLASSES[i]}</span>'
                  f'<span>{probs[i]*100:.1f}%</span></div><div class="track"><div class="fill" style="width:{probs[i]*100:.1f}%"></div></div></div>')
    st.markdown(html_, unsafe_allow_html=True)
with cr:
    st.markdown("### Reliability checks")
    st.markdown("".join(f'<div class="chk"><i style="background:{"var(--ok)" if ok else "var(--warn)"}"></i><span>{t_}</span></div>'
                        for ok, t_ in checks), unsafe_allow_html=True)

# ───────────────────────── EXPORT ─────────────────────────
st.markdown("### Export")
settings_txt = (f"decision threshold {DEC_THR:.2f}; overlay threshold {seg_thr:.2f}; min lesion {MIN_LESION_PCT:.2f}%; "
                f"fragment mean/peak {COMP_MEAN:.2f}/{COMP_PEAK:.2f}; auto-crop {'on' if crop else 'off'}; "
                f"intensity alignment {'on' if robust else 'off'}; "
                f"multi-view {'on' if multiview else 'off'} ({n_views} views); temperature {T:.2f}")
sha = hashlib.sha1(img_bytes).hexdigest()[:10]
report = {
    "result": final_name, "decision_state": state, "confidence": round(final_conf, 4), "confidence_is": conf_label,
    "raw_prediction": CLASSES[pred], "heads_disagree": bool(conflict), "reliability": level, "view_agreement": round(agree, 3),
    "lesion_area_percent": round(area, 3), "mask_removed_as_false_positive": state == "fp", "segmentation_view_agreement": round(seg_cons, 3), "lesion_confidence": round(les_conf, 3),
    "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
    "tumour_type_ranking": {n: round(p, 4) for n, p in type_rank},
    "settings": settings_txt, "input_warnings": ood, "image_sha1": sha,
}
png = png_bytes(np.hstack([titled(p, c) for p, c in zip(panels, captions)]))

e1, e2 = st.columns(2)
stretch(e1.download_button, "Download prediction report (JSON)", json.dumps(report, indent=2), "prediction_report.json", "application/json")
stretch(e2.download_button, "Download visualization (PNG)", png, "visualization.png", "image/png")

st.markdown("##### Full report (PDF)")
st.caption("Fill in the details you want printed. Leave a field empty to print a dash. Use English letters for names.")
c1, c2, c3 = st.columns(3)
patient = c1.text_input("Patient name")
pid = c2.text_input("Patient ID")
agesex = c3.text_input("Age / Sex", placeholder="e.g. 45 / M")
d1, d2, d3 = st.columns(3)
doctor = d1.text_input("Doctor name", placeholder="Dr. ...")
desig = d2.text_input("Designation / Reg. no.")
facility = d3.text_input("Hospital / Clinic")
notes = st.text_area("Clinical notes (optional)", height=80)

doctor, desig, facility = (doctor.strip() or def_doctor.strip(), desig.strip() or def_desig.strip(),
                           facility.strip() or def_facility.strip())
missing = [n for n, v in [("Patient name", patient), ("Patient ID", pid), ("Age / Sex", agesex),
                          ("Doctor name", doctor), ("Designation", desig), ("Hospital / Clinic", facility)] if not v.strip()]
if missing:
    st.warning("Not filled yet (will print as 'Not provided'): " + ", ".join(missing) + ". "
               "Doctor, designation and hospital can be saved once under Report defaults in the sidebar.")
else:
    st.success("All report fields are filled.")
if any(ord(ch) > 255 for ch in patient + doctor + facility + notes):
    st.warning("Some characters are not Latin (e.g. Bengali letters) and will print as '?' in the PDF. Write names in English letters.")

if HAS_PDF:
    info = dict(patient=patient, pid=pid, agesex=agesex, doctor=doctor, desig=desig, facility=facility, notes=notes,
                date=datetime.now(TZ).strftime("%d %b %Y, %I:%M %p"))
    r = dict(name=final_name, conf=final_conf, area=area, level=level, summary=summary, title=title, probs=probs, pred=pred,
             checks=checks, settings=settings_txt, sha=sha, rid=f"MRI-{sha.upper()}",
             types=type_rank if state in ("tumour", "conflict", "no_mask") else [], conf_label=conf_label)
    pdf = build_pdf(info, r, [png_bytes(p, 448) for p in panels], captions)
    stretch(st.download_button, "Download full report (PDF)", pdf, f"MRI_report_{sha}.pdf", "application/pdf", type="primary")
else:
    st.info("PDF export needs `reportlab`. Add it to requirements.txt and redeploy.")

st.markdown('<div class="note">Research prototype for a thesis project. Not a medical device and not for clinical decisions.</div>', unsafe_allow_html=True)
