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
import hashlib
from dataclasses import dataclass, field

OUTPUT_ROOT = "output"


# ---- data model ------------------------------------------------------------

@dataclass
class Block:
    label: str
    content: str
    page: int
    order: int


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
        """Flat list of (page_no, order, html) for every table block."""
        out = []
        for pg in self.pages:
            for b in pg.blocks:
                if b.label == "table" and b.content:
                    out.append((pg.page_no, b.order, b.content))
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


def load_pages_from_dir(out_dir):
    """Load page_*.json (PPStructureV3 save_to_json format) into Page objects."""
    paths = sorted(glob.glob(os.path.join(out_dir, "page_*.json")), key=_page_num)
    pages = []
    for path in paths:
        with open(path, "r", encoding="utf8") as f:
            d = json.load(f)
        blocks = []
        for b in d.get("parsing_res_list", []):
            blocks.append(
                Block(
                    label=b.get("block_label", ""),
                    content=b.get("block_content", ""),
                    page=_page_num(path),
                    order=(
                        b.get("block_order")
                        if b.get("block_order") is not None
                        else b.get("block_id", 0)
                    ),
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
    return bool(glob.glob(os.path.join(out_dir, "page_*.json")))


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

# Fast = mobile models (default weights). Accurate = server-tier + deskew.
TIER_KWARGS = {
    "fast": {},
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


def _build_pipeline(tier, device):
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

    sig = inspect.signature(PPStructureV3.__init__).parameters
    has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.values()
    )
    kwargs = {
        k: v
        for k, v in candidate.items()
        if v is not None and (has_var_kwargs or k in sig)
    }
    return PPStructureV3(**kwargs)


def extract(pdf_path, tier="fast", device="cpu", page_range=None, progress=None):
    """
    Return an Extraction for pdf_path. Uses cache when available.

    progress: optional callable(done, total, msg) for UI progress bars.
    page_range: optional (start, end) 1-indexed inclusive; None = whole doc.
    """
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
    doc_type = doc_type_from_pdf(pdf_path) or "Scanned"
    pipeline = _build_pipeline(tier, device)

    page_count = 0
    for result in pipeline.predict_iter(pdf_path):
        page_count += 1
        if page_range and not (page_range[0] <= page_count <= page_range[1]):
            continue
        result.save_to_json(os.path.join(out_dir, f"page_{page_count}.json"))
        if progress:
            progress(page_count, None, f"OCR page {page_count}")

    write_meta(out_dir, doc_type=doc_type, tier=tier)
    pages = load_pages_from_dir(out_dir)
    return Extraction(
        pages=pages, from_cache=False, out_dir=out_dir, tier=tier,
        doc_type=doc_type,
    )


def _guess_doc_type(pages):
    """Fallback when no PDF text-layer signal is available (e.g. legacy cache
    dir): many pages carrying full-page image blocks => scanned."""
    if not pages:
        return "Scanned"
    with_img = sum(1 for pg in pages if any(b.label == "image" for b in pg.blocks))
    return "Scanned" if with_img >= 0.4 * len(pages) else "Digital"
