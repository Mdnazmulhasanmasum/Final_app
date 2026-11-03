import io, json, time, hashlib
from html import escape as esc
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, ImageOps

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
MIN_LESION_PCT = 0.40
TZ = timezone(timedelta(hours=6))  # Bangladesh time

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


def run_once(t):
    res = sess.run(OUT_NAMES, {IN_NAME: t})
    cls = np.squeeze(res[0]).astype(np.float64)
    seg = np.squeeze(res[1]).astype(np.float32) if len(res) > 1 else None
    attr = np.squeeze(res[2]).astype(np.float32) if len(res) > 2 else None
    return cls, seg, attr


@st.cache_data(show_spinner="Analysing scan…", max_entries=24)
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
    return rgb, np.stack(logits), seg, attr, (time.perf_counter() - t0) * 1000


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
    S = lambda n, **k: ParagraphStyle(n, parent=ss["Normal"], textColor=ink, **k)
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
             P(f"TinyBrainNet-MT &middot; Khwaja Yunus Ali University &middot; Report ID {r['rid']}", sub),
             Spacer(1, 3), HRFlowable(width="100%", thickness=1.2, color=acc, spaceAfter=8)]

    cw = [27 * mm, W / 2 - 27 * mm] * 2
    rows = [[P("Patient name", lab), P(latin(info["patient"]) or "-", val), P("Patient ID", lab), P(latin(info["pid"]) or "-", val)],
            [P("Age / Sex", lab), P(latin(info["agesex"]) or "-", val), P("Report date", lab), P(info["date"], val)],
            [P("Doctor", lab), P(latin(info["doctor"]) or "-", val), P("Designation", lab), P(latin(info["desig"]) or "-", val)],
            [P("Facility", lab), P(latin(info["facility"]) or "-", val), P("Scan file", lab), P(r["sha"], val)]]
    t = Table(rows, colWidths=cw)
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, line), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LEFTPADDING", (0, 0), (-1, -1), 2)]))
    story += [t, P("AI findings", h2)]

    lv = {"High": "#16a34a", "Medium": "#d97706", "Low": "#dc2626"}[r["level"]]
    res = Table([[P("Result", lab), P("Confidence", lab), P("Lesion area", lab), P("Reliability", lab)],
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
                       help="Stops the model from showing 100% on scans it has never seen the like of.")
    seg_thr = st.slider("Segmentation threshold", 0.10, 0.90, 0.50, 0.05)
    alpha = st.slider("Overlay strength", 0.1, 0.9, 0.50, 0.05)
    st.divider()
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

# ───────────────────────── INFERENCE ─────────────────────────
try:
    disp, logits, seg_raw, attr_raw, ms = infer(img_bytes, crop, robust, multiview)
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

if seg_raw is not None:
    sp = seg_raw if SEG_IS_PROB else sigmoid(seg_raw)
    sp = cv2.resize(sp.astype(np.float32), (IMG, IMG), interpolation=cv2.INTER_LINEAR)
    mask = sp >= seg_thr
else:
    sp = np.zeros((IMG, IMG), np.float32)
    mask = np.zeros((IMG, IMG), bool)
area = float(mask.mean()) * 100
les_conf = float(sp[mask].mean()) if mask.any() else 0.0
has_lesion = area >= MIN_LESION_PCT

pred_is_clear = NO_TUMOR is not None and pred == NO_TUMOR
conflict = pred_is_clear and has_lesion                     # classifier: no tumour / segmentation: mass
empty_seg = (not pred_is_clear) and not has_lesion          # classifier: tumour / segmentation: nothing

# Tumour type, ranked among tumour classes only. When the heads disagree the scan is probably out of
# distribution, so the type is softened harder and shown as a suggestion rather than a verdict.
T_type = max(T, 2.0) if conflict else T
tp = softmax(mean_logits[TUMOR_IDX], T_type)
order = np.argsort(-tp)
n1, p1 = CLASSES[TUMOR_IDX[order[0]]], float(tp[order[0]])
n2, p2 = (CLASSES[TUMOR_IDX[order[1]]], float(tp[order[1]])) if len(order) > 1 else ("", 0.0)
type_rank = [(CLASSES[TUMOR_IDX[j]], float(tp[j])) for j in order]

if conflict:
    final_idx, final_conf = TUMOR_IDX[order[0]], p1
else:
    final_idx, final_conf = pred, conf
final_name = CLASSES[final_idx]

# Reliability
heads_ok = not conflict and not empty_seg
if conflict:
    level = "Low"
else:
    score = 0.5 * conf + 0.5 * agree
    level = "High" if score >= 0.85 and heads_ok else "Medium" if score >= 0.60 else "Low"
    if not heads_ok and level == "High":
        level = "Medium"

checks = [(agree >= 0.8, f"{int(round(agree * n_views))} of {n_views} views give the same class.")]
if conflict:
    checks.append((False, f"Classifier says No_tumor but segmentation outlined {area:.2f}% of the slice."))
elif empty_seg:
    checks.append((False, f"Classifier says {CLASSES[pred]} but segmentation found no clear lesion."))
else:
    checks.append((True, "Classifier and segmentation agree."))
if has_lesion:
    checks.append((les_conf >= 0.70, f"Lesion confidence (mean mask probability) {les_conf*100:.0f}%."))
if not multiview:
    checks.append((False, "Multi-view check is off. Turn it on for a steadier result."))
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
if conflict:
    title = "Lesion found, tumour type uncertain"
    summary = (f"The segmentation head outlined a mass covering <b>{area:.2f}%</b> of the slice, but the classifier answered "
               f"<b>No_tumor</b>. This usually means the scan looks different from the training data (watermark, contrast, "
               f"scanner or view). Treat the type as a suggestion: most likely <b>{n1}</b> ({p1*100:.0f}%), then {n2} ({p2*100:.0f}%). "
               f"Please review manually.")
elif empty_seg:
    title = f"{final_name} predicted, no clear lesion outlined"
    summary = (f"The classifier favours <b>{final_name}</b> ({conf*100:.1f}%), but the segmentation head did not outline a lesion "
               f"above the {seg_thr:.2f} threshold. Check the scan manually.")
elif pred_is_clear:
    title = "No tumour detected"
    summary = f"Both heads agree. <b>No_tumor</b> at {conf*100:.1f}% confidence, no lesion outlined."
else:
    title = f"{final_name} detected"
    summary = (f"Both heads agree. <b>{final_name}</b> at {conf*100:.1f}% confidence, lesion covers <b>{area:.2f}%</b> of the slice."
               + ("" if level == "High" else " Reliability is not high, so review before relying on it."))

tone = "var(--ok)" if level == "High" else "var(--warn)"
st.markdown(f'<div class="banner" style="--c:{tone}"><span class="tt">{title}</span><div>{summary}</div></div>', unsafe_allow_html=True)

lvl_col = {"High": "var(--ok)", "Medium": "var(--warn)", "Low": "var(--bad)"}[level]
k = st.columns(4)
for col, lab_, v_, sty in [(k[0], "Likely type" if conflict else "Diagnosis", final_name, ""),
                           (k[1], "Type confidence" if conflict else "Confidence", f"{final_conf*100:.1f}%", ""),
                           (k[2], "Lesion area", f"{area:.2f}%", ""),
                           (k[3], "Reliability", level, f"color:{lvl_col}")]:
    col.markdown(f'<div class="kpi"><div class="l">{lab_}</div><div class="v" style="{sty}">{v_}</div></div>', unsafe_allow_html=True)

# ───────────────────────── VISUAL INSPECTION ─────────────────────────
st.markdown("### Visual inspection")
overlay = blend_mask(disp, mask, alpha) if mask.any() else disp
heat = blend_heat(disp, attr, alpha) if attr is not None else None
panels = [disp, overlay] + ([heat] if heat is not None else [])
captions = ["Resized model input", f"Predicted mask · threshold {seg_thr:.2f}"] + (["Model attribution (experimental)"] if heat is not None else [])
for col, img_, cap_ in zip(st.columns(len(panels), gap="medium"), panels, captions):
    col.markdown(f'<div class="cap">{cap_}</div>', unsafe_allow_html=True)
    col.image(img_, use_container_width=True)

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
settings_txt = (f"threshold {seg_thr:.2f}; auto-crop {'on' if crop else 'off'}; intensity alignment {'on' if robust else 'off'}; "
                f"multi-view {'on' if multiview else 'off'} ({n_views} views); temperature {T:.2f}")
sha = hashlib.sha1(img_bytes).hexdigest()[:10]
report = {
    "diagnosis": final_name, "confidence": round(final_conf, 4), "raw_prediction": CLASSES[pred],
    "heads_disagree": bool(conflict), "reliability": level, "view_agreement": round(agree, 3),
    "lesion_area_percent": round(area, 3), "lesion_confidence": round(les_conf, 3),
    "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
    "tumour_type_ranking": {n: round(p, 4) for n, p in type_rank},
    "settings": settings_txt, "image_sha1": sha,
}
png = png_bytes(np.hstack([titled(p, c) for p, c in zip(panels, captions)]))

e1, e2 = st.columns(2)
e1.download_button("Download prediction report (JSON)", json.dumps(report, indent=2), "prediction_report.json", "application/json", use_container_width=True)
e2.download_button("Download visualization (PNG)", png, "visualization.png", "image/png", use_container_width=True)

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

if HAS_PDF:
    info = dict(patient=patient, pid=pid, agesex=agesex, doctor=doctor, desig=desig, facility=facility, notes=notes,
                date=datetime.now(TZ).strftime("%d %b %Y, %I:%M %p"))
    r = dict(name=final_name, conf=final_conf, area=area, level=level, summary=summary, title=title, probs=probs, pred=pred,
             checks=checks, settings=settings_txt, sha=sha, rid=f"MRI-{sha.upper()}",
             types=type_rank if (final_idx != NO_TUMOR) else [])
    pdf = build_pdf(info, r, [png_bytes(p, 448) for p in panels], captions)
    st.download_button("Download full report (PDF)", pdf, f"MRI_report_{sha}.pdf", "application/pdf", use_container_width=True, type="primary")
else:
    st.info("PDF export needs `reportlab`. Add it to requirements.txt and redeploy.")

st.markdown('<div class="note">Research prototype for a thesis project. Not a medical device and not for clinical decisions.</div>', unsafe_allow_html=True)
