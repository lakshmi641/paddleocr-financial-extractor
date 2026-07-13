"""
exports.py

Turn an Extraction into downloadable artifacts: Excel (one sheet per table),
JSON (text + tables), and Markdown. In-memory bytes for Streamlit download
buttons, plus a write-to-disk helper for the auto-save banner.
"""

import io
import os
import json

import pandas as pd

from classify import html_to_df


def build_excel(extraction):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        wrote_any = False
        for page_no, order, html in extraction.tables:
            df = html_to_df(html)
            if df is None:
                continue
            sheet = f"P{page_no}_{order}"[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)
            wrote_any = True
        if not wrote_any:
            pd.DataFrame({"Info": ["No tables detected"]}).to_excel(
                writer, index=False
            )
    buf.seek(0)
    return buf.getvalue()


def build_json(extraction, statements=None):
    data = {
        "doc_type": extraction.doc_type,
        "tier": extraction.tier,
        "page_count": len(extraction.pages),
        "full_text": extraction.full_text,
        "tables": [],
    }
    for page_no, order, html in extraction.tables:
        df = html_to_df(html)
        data["tables"].append(
            {
                "page": page_no,
                "order": order,
                "rows": [] if df is None else df.fillna("").to_dict("records"),
            }
        )
    if statements:
        data["core_statements"] = {
            k: {
                "page": s.page_no,
                "reconciliation": (
                    None if s.recon is None
                    else {"status": s.recon.status, "detail": s.recon.detail}
                ),
            }
            for k, s in statements.items()
            if s.page_no is not None
        }
    return json.dumps(data, indent=2, ensure_ascii=False).encode("utf8")


def build_markdown(extraction):
    lines = [f"# Extracted Document ({extraction.doc_type})\n"]
    for pg in extraction.pages:
        lines.append(f"\n<!-- Page {pg.page_no} -->\n")
        for b in pg.blocks:
            if b.label == "table" and b.content:
                df = html_to_df(b.content)
                if df is not None:
                    lines.append(df.to_markdown(index=False) + "\n")
                continue
            if b.label in ("image", "seal"):
                continue
            text = " ".join(str(b.content).split())
            if text:
                lines.append(text + "\n")
    return "\n".join(lines).encode("utf8")


def write_outputs(extraction, statements=None):
    """Auto-save the three artifacts into the extraction's cache dir.
    Returns the out_dir for the banner."""
    out_dir = extraction.out_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "document.xlsx"), "wb") as f:
        f.write(build_excel(extraction))
    with open(os.path.join(out_dir, "document.json"), "wb") as f:
        f.write(build_json(extraction, statements))
    with open(os.path.join(out_dir, "document.md"), "wb") as f:
        f.write(build_markdown(extraction))
    return out_dir
