"""
ocr_engine.py

PDF -> PaddleOCR (PPStructureV3) -> normalized page blocks, with an on-disk cache.

The heavy paddleocr import is deferred until we actually need to OCR a new document,
so the Streamlit app (and the pure classify/reconcile modules) can be imported and
tested without paddle installed as long as everything hits the cache.
"""

import os
import re
import glob
import json
import shutil
import hashlib
from dataclasses import dataclass, field

# oneDNN's PIR executor raises NotImplementedError on some layout/table models
# with this paddlepaddle build (ConvertPirAttribute2RuntimeAttribute on
# ArrayAttribute<DoubleAttribute>). Must be set before paddlex is imported.
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")

OUTPUT_ROOT = "output"

# Overridable so the same code runs unchanged on a CPU dev machine and the
# GPU (T4) VM: set OCR_DEVICE=gpu:0 there. OCR_PRECISION=fp16 is the usual
# pairing for a T4 (halves VRAM, minimal accuracy loss) but only takes
# effect on GPU — PaddleOCR ignores it on CPU.
DEFAULT_DEVICE = os.environ.get("OCR_DEVICE", "cpu")
DEFAULT_PRECISION = os.environ.get("OCR_PRECISION")

# Same env var / default paddlex itself reads (paddlex.utils.flags.PDF_RENDER_SCALE)
# to rasterize PDF pages before OCR. _extract_auto re-renders individual pages
# itself (see below) and must match this scale, or its "accurate" re-OCR of a
# page runs at a different resolution than a full "accurate" pass would have
# used -- which measurably shifts how the layout/table models segment that
# page (verified: mismatched table counts on 4 pages before this was fixed).
PDF_RENDER_SCALE = float(os.environ.get("PADDLE_PDX_PDF_RENDER_SCALE", "2.0"))


# ---- data model ------------------------------------------------------------

@dataclass
class Block:
    label: str
    content: str
    page: int
    order: int
    # For table blocks only: raw OCR words (text + bbox) underneath the
    # table, straight from PPStructureV3's table_ocr_pred. The table
    # *structure* model occasionally mis-predicts the cell grid on dense
    # tables (merges dozens of rows into one giant cell) even though the
    # underlying word-level OCR is fine; table_repair.py uses these to
    # rebuild a sane grid when that happens. None for non-table blocks.
    table_words: list = None


@dataclass
class Page:
    page_no: int
    width: int
    height: int
    blocks: list = field(default_factory=list)


@dataclass
class Extraction:
    pages: list = field(default_factory=list)
    doc_type: str = "Scanned"
    from_cache: bool = False
    out_dir: str = ""
    tier: str = "fast"

    @property
    def tables(self):
        """Flat list of (page_no, order, html, table_words) for every table
        block. table_words is the raw OCR word/box list backing that table
        (see Block.table_words), or None if unavailable."""
        out = []
        for pg in self.pages:
            for b in pg.blocks:
                if b.label == "table" and b.content:
                    out.append((pg.page_no, b.order, b.content, b.table_words))
        return out

    @property
    def full_text(self):
        parts = []
        for pg in self.pages:
            parts.append(f"\n===== PAGE {pg.page_no} =====\n")
            for b in pg.blocks:
                if b.label in ("table", "image", "seal"):
                    continue
                text = " ".join(str(b.content).split())
                if text:
                    parts.append(text)
        return "\n".join(parts)


# ---- cache helpers ---------------------------------------------------------

def pdf_hash(pdf_bytes):
    return hashlib.md5(pdf_bytes).hexdigest()[:8]


def cache_dir_for(stem, digest, tier):
    return os.path.join(OUTPUT_ROOT, f"{stem}_{tier}_{digest}")


def _page_num(path):
    m = re.search(r"page_(\d+)\.json$", path)
    return int(m.group(1)) if m else 0


def _table_words_list(page_json):
    """Extract (text, box) pairs per table from PPStructureV3's table_res_list,
    in the same left-to-right/top-to-bottom order the tables themselves appear
    on the page. Returns a list (one entry per table on the page) of word-lists."""
    out = []
    for tbl in page_json.get("table_res_list", []) or []:
        pred = tbl.get("table_ocr_pred") or {}
        texts = pred.get("rec_texts") or []
        boxes = pred.get("rec_boxes") or []
        words = [
            (str(t), tuple(b))
            for t, b in zip(texts, boxes)
            if not str(t).strip().startswith("<div")  # embedded image placeholder
        ]
        out.append(words)
    return out


def load_pages_from_dir(out_dir):
    """Load page_*.json (PPStructureV3 save_to_json format) into Page objects."""
    paths = sorted(glob.glob(os.path.join(out_dir, "page_*.json")), key=_page_num)
    pages = []
    for path in paths:
        with open(path, "r", encoding="utf8") as f:
            d = json.load(f)
        table_words = _table_words_list(d)
        table_i = 0
        blocks = []
        for b in d.get("parsing_res_list", []):
            label = b.get("block_label", "")
            words = None
            if label == "table":
                if table_i < len(table_words):
                    words = table_words[table_i]
                table_i += 1
            blocks.append(
                Block(
                    label=label,
                    content=b.get("block_content", ""),
                    page=_page_num(path),
                    order=(
                        b.get("block_order")
                        if b.get("block_order") is not None
                        else b.get("block_id", 0)
                    ),
                    table_words=words,
                )
            )
        blocks.sort(key=lambda x: x.order if x.order is not None else 10_000)
        pages.append(
            Page(
                page_no=_page_num(path),
                width=d.get("width", 0),
                height=d.get("height", 0),
                blocks=blocks,
            )
        )
    return pages


def has_cache(out_dir):
    """A cache is only valid if meta.json exists: it's written exclusively
    after a full extract() loop completes, so its presence rules out a
    partial cache left behind by a crashed/interrupted run."""
    return os.path.exists(_meta_path(out_dir)) and bool(
        glob.glob(os.path.join(out_dir, "page_*.json"))
    )


def _clear_partial(out_dir):
    """Remove any page_*.json/meta.json left behind by a prior run that
    never finished, so a crash can't leave a stale partial cache on disk."""
    for path in glob.glob(os.path.join(out_dir, "page_*.json")):
        os.remove(path)
    meta_path = _meta_path(out_dir)
    if os.path.exists(meta_path):
        os.remove(meta_path)


def _meta_path(out_dir):
    return os.path.join(out_dir, "meta.json")


def read_meta(out_dir):
    try:
        with open(_meta_path(out_dir), "r", encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return {}


def write_meta(out_dir, **kv):
    with open(_meta_path(out_dir), "w", encoding="utf8") as f:
        json.dump(kv, f, indent=2)


def doc_type_from_pdf(pdf_path):
    """Definitive signal: a PDF with no extractable text layer is a scan.
    Requires PyMuPDF; returns None if unavailable so callers fall back."""
    try:
        import fitz  # PyMuPDF
    except Exception:
        return None
    try:
        doc = fitz.open(pdf_path)
        chars = sum(len(doc[i].get_text()) for i in range(len(doc)))
        return "Scanned" if chars < 100 else "Digital"
    except Exception:
        return None


# ---- OCR --------------------------------------------------------------------

# Fast = mobile models (lighter/quicker). Accurate = server-tier + deskew.
# "auto" isn't a pipeline config -- it's handled in _extract_auto, which
# combines these two per page instead of picking one for the whole document.
# NOTE: PPStructureV3's own defaults are the SERVER det/rec models, so "fast"
# must explicitly request the mobile ones or it silently runs just as slow
# as "accurate".
TIER_KWARGS = {
    "fast": {
        "text_detection_model_name": "PP-OCRv5_mobile_det",
        "text_recognition_model_name": "PP-OCRv5_mobile_rec",
    },
    "accurate": {
        "text_detection_model_name": "PP-OCRv5_server_det",
        "text_recognition_model_name": "PP-OCRv5_server_rec",
        "wired_table_structure_recognition_model_name": "SLANet_plus",
        "use_doc_orientation_classify": True,
        "use_doc_unwarping": True,
        "text_det_limit_side_len": 4000,
        "text_det_limit_type": "max",
    },
}


_PIPELINE_CACHE = {}


def _build_pipeline(tier, device):
    """Building a PPStructureV3 pipeline loads ~13 CPU models and is by far
    the most expensive part of extraction (minutes, vs. seconds/page once
    warm) — so keep one pipeline per (tier, device) alive for the life of
    the process instead of rebuilding it on every extract() call."""
    cached = _PIPELINE_CACHE.get((tier, device))
    if cached is not None:
        return cached

    import inspect
    from paddleocr import PPStructureV3

    candidate = {
        "device": device,
        "use_formula_recognition": False,
        "use_seal_recognition": False,
        "use_chart_recognition": False,
    }
    candidate.update(TIER_KWARGS.get(tier, {}))
    if device == "cpu":
        candidate["cpu_threads"] = int(os.environ.get("OCR_CPU_THREADS", "8"))
    else:
        candidate["precision"] = DEFAULT_PRECISION

    sig = inspect.signature(PPStructureV3.__init__).parameters
    has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.values()
    )
    kwargs = {
        k: v
        for k, v in candidate.items()
        if v is not None and (has_var_kwargs or k in sig)
    }
    pipeline = PPStructureV3(**kwargs)
    _PIPELINE_CACHE[(tier, device)] = pipeline
    return pipeline


def extract(pdf_path, tier="fast", device=None, page_range=None, progress=None):
    """
    Return an Extraction for pdf_path. Uses cache when available.

    progress: optional callable(done, total, msg) for UI progress bars.
    page_range: optional (start, end) 1-indexed inclusive; None = whole doc.
    device: "cpu", "gpu", "gpu:0", etc. Defaults to OCR_DEVICE env var (see
        DEFAULT_DEVICE) so the same call runs on CPU locally and GPU on the VM
        without callers having to know which machine they're on.
    tier: "fast" (mobile models, quick), "accurate" (server models, slower,
        sharper on small/dense digits), or "auto" -- see _extract_auto.
    """
    if device is None:
        device = DEFAULT_DEVICE
    if tier == "auto":
        return _extract_auto(pdf_path, device, page_range, progress)
    with open(pdf_path, "rb") as f:
        digest = pdf_hash(f.read())
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    out_dir = cache_dir_for(stem, digest, tier)

    if has_cache(out_dir):
        pages = load_pages_from_dir(out_dir)
        doc_type = read_meta(out_dir).get("doc_type") or _guess_doc_type(pages)
        return Extraction(
            pages=pages, from_cache=True, out_dir=out_dir, tier=tier,
            doc_type=doc_type,
        )

    os.makedirs(out_dir, exist_ok=True)
    _clear_partial(out_dir)  # drop any leftovers from a previously crashed run
    doc_type = doc_type_from_pdf(pdf_path) or "Scanned"
    pipeline = _build_pipeline(tier, device)

    page_count = 0
    try:
        for result in pipeline.predict_iter(pdf_path):
            page_count += 1
            if page_range and not (page_range[0] <= page_count <= page_range[1]):
                continue
            result.save_to_json(os.path.join(out_dir, f"page_{page_count}.json"))
            if progress:
                progress(page_count, None, f"OCR page {page_count}")
    except Exception:
        _clear_partial(out_dir)
        raise

    write_meta(out_dir, doc_type=doc_type, tier=tier)
    pages = load_pages_from_dir(out_dir)
    return Extraction(
        pages=pages, from_cache=False, out_dir=out_dir, tier=tier,
        doc_type=doc_type,
    )


def _extract_auto(pdf_path, device, page_range, progress):
    """Hybrid tier: one fixed tier for the whole document is a blunt choice --
    "fast" blurs small digits on dense tables, "accurate" pays server-model
    cost on cover/signature pages that never had a table to begin with. So:
    OCR every page with "fast" first, then re-OCR (at "accurate") only the
    pages that came back with a table block -- the pages a reconciliation
    actually reads numbers from. Narrative pages keep their fast-tier result.
    """
    with open(pdf_path, "rb") as f:
        digest = pdf_hash(f.read())
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    out_dir = cache_dir_for(stem, digest, "auto")

    if has_cache(out_dir):
        pages = load_pages_from_dir(out_dir)
        doc_type = read_meta(out_dir).get("doc_type") or _guess_doc_type(pages)
        return Extraction(
            pages=pages, from_cache=True, out_dir=out_dir, tier="auto",
            doc_type=doc_type,
        )

    fast = extract(pdf_path, tier="fast", device=device, page_range=page_range,
                    progress=progress)

    os.makedirs(out_dir, exist_ok=True)
    _clear_partial(out_dir)
    for path in glob.glob(os.path.join(fast.out_dir, "page_*.json")):
        shutil.copy(path, os.path.join(out_dir, os.path.basename(path)))

    upgrade_pages = sorted(
        pg.page_no for pg in fast.pages if any(b.label == "table" for b in pg.blocks)
    )

    if upgrade_pages:
        import pypdfium2 as pdfium

        # Use the same rasterizer (and scale) paddlex's own PDF reader uses,
        # not PyMuPDF/fitz: a different renderer produces different
        # anti-aliasing at the pixel level even at matched nominal DPI, which
        # measurably changes recognition on borderline characters (verified:
        # it flipped one word's OCR, silently breaking a keyword match in
        # reconcile.py). pdfium's render() already returns BGR, so no channel
        # swap is needed (unlike fitz, which renders RGB).
        pdf_doc = pdfium.PdfDocument(pdf_path)
        pipeline = _build_pipeline("accurate", device)
        for i, page_no in enumerate(upgrade_pages, 1):
            if progress:
                progress(
                    len(fast.pages) + i, None,
                    f"Upgrading table page {page_no} ({i}/{len(upgrade_pages)})",
                )
            img = pdf_doc[page_no - 1].render(scale=PDF_RENDER_SCALE).to_numpy()
            result = next(iter(pipeline.predict(img)))
            result.save_to_json(os.path.join(out_dir, f"page_{page_no}.json"))

    write_meta(
        out_dir, doc_type=fast.doc_type, tier="auto", upgraded_pages=upgrade_pages,
    )
    pages = load_pages_from_dir(out_dir)
    return Extraction(
        pages=pages, from_cache=False, out_dir=out_dir, tier="auto",
        doc_type=fast.doc_type,
    )


def _guess_doc_type(pages):
    """Fallback when no PDF text-layer signal is available (e.g. legacy cache
    dir): many pages carrying full-page image blocks => scanned."""
    if not pages:
        return "Scanned"
    with_img = sum(1 for pg in pages if any(b.label == "image" for b in pg.blocks))
    return "Scanned" if with_img >= 0.4 * len(pages) else "Digital"
