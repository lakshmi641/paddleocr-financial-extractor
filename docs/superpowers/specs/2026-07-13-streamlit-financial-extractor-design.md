# Streamlit Financial Statement Extractor (PaddleOCR) — Design

Date: 2026-07-13

## Goal
A Streamlit UI over the existing PaddleOCR (PPStructureV3) pipeline that mirrors the
"Financial Statement Extractor" layout: upload a financial PDF, extract tables + text,
auto-classify Balance Sheet / P&L / Cash Flow, best-effort reconcile the numbers, and
export Excel/JSON/Markdown. Runs locally on CPU, reuses an on-disk cache.

## Decisions (from brainstorming)
- **Scope:** full parity — classification + best-effort reconciliation + all tables + full text.
- **Compute:** local CPU; reuse `output/<stem>_<hash>/` cache; first-run OCR only.
- **Engine:** PaddleOCR only; Fast (mobile models) ↔ Accurate (server models) toggle
  (maps to the model-tier env work in `extract.py`). No multi-engine dropdown, no cross-check.

## Modules (single responsibility each)
- `app.py` — Streamlit UI only (layout, widgets, rendering). No OCR/parse logic.
- `ocr_engine.py` — `extract(pdf_path, tier, scope) -> Extraction`. Runs PPStructureV3 or
  loads cache. Owns the md5-based cache. Returns normalized page blocks.
- `classify.py` — `classify(pages) -> {balance_sheet, profit_loss, cash_flow, other}`.
  Keyword-scoring over table header cells + surrounding text.
- `reconcile.py` — `check(statement) -> ReconResult(status, detail)`; status ∈
  {ok, could_not_verify}. Balance Sheet: Total Assets ≈ Equity + Liabilities.
  Cash Flow: opening + net change ≈ closing.
- `exports.py` — `write_outputs(extraction, out_dir)` → Excel + JSON + Markdown.
  Refactored from `convert.py` logic (one source of truth).

`extract.py` / `convert.py` (CLI path) remain; the new modules reuse their logic.

## Data model
```
Block   = {label, content, page, order}         # label: text|table|doc_title|... ; content: str/HTML
Page    = {page_no, width, height, blocks:[Block]}
Table   = {page_no, df: DataFrame, html, caption}
Extraction = {pages:[Page], tables:[Table], full_text:str, doc_type:"Scanned"|"Digital",
              from_cache:bool, out_dir:str, tier:str}
Statement = {name, page_no, tables:[Table], recon: ReconResult}
```

## Data flow
1. Upload → md5(pdf bytes). Cache dir `output/<stem>_<hash8>/` (+ tier in key).
2. Cache hit → load page_*.json (banner "⚡ Loaded from cache"). Miss → run PaddleOCR,
   write page_*.json.
3. `classify` → assign statement pages; highest-scoring statement wins each table page.
4. `reconcile` each core statement (best-effort).
5. `exports.write_outputs` → auto-save banner "💾 Auto-saved to <dir> (Excel + JSON + Markdown)".
6. Render metrics + tabs.

## Classification heuristic (validated on sample doc)
Score each page by keyword hits, weight table-bearing pages higher.
- Balance Sheet: `balance sheet`, `equity and liabilities`, `total assets`, `total equity`
- P&L: `profit and loss`, `revenue from operations`, `total income`, `profit for the year`
- Cash Flow: `cash flow`, `operating activities`, `investing activities`, `financing activities`
Verified: sample doc resolves core statements to pages 12 (BS), 13 (P&L), 14 (CF).

## Reconciliation — honest limits
Best-effort numeric parsing from OCR'd tables (merged cells, Rs. Lakhs). Reviewer aid,
NOT a guarantee. When totals can't be parsed/aligned → yellow "Could not verify —
please review" (never a false green check).

## UI layout (mirrors reference screenshot)
- Sidebar **Settings**: engine=PaddleOCR (static), Scope (Whole doc / Page range),
  Fast mode toggle, first-run caption.
- Main: title + description, PDF uploader, **Extract** button, 4-metric row
  (Document type · Engine · Core statements · Tables extracted), cache banner,
  auto-saved banner, tabs **Balance Sheet · Profit & Loss · Cash Flow · All tables ·
  Full text**, download row (Excel/JSON/MD).

## Testing
- `classify` and `reconcile` are pure functions over the cached JSON → unit-testable
  without Streamlit/GPU using the existing `output/page_*.json` as fixtures.

## Out of scope
Multi-engine cross-check, GPU/VM serving (kept swappable via `ocr_engine`), training,
non-financial doc types.
