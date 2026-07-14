"""
Financial Statement Extractor — Streamlit UI over the PaddleOCR pipeline.

Run:  streamlit run app.py
"""

import os
import tempfile

import streamlit as st

import ocr_engine
import classify as clf
import reconcile as rec
import exports

st.set_page_config(
    page_title="Financial Statement Extractor",
    page_icon="📊",
    layout="wide",
)

# ---- light styling ---------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 2.5rem; max-width: 1150px;}
      div[data-testid="stMetricValue"] {font-size: 2.1rem;}
      .stTabs [data-baseweb="tab-list"] {gap: 1.5rem;}
      .stTabs [data-baseweb="tab"] {font-size: 1rem; font-weight: 600;}
      .banner {padding: 0.85rem 1.1rem; border-radius: 0.6rem; margin: 0.4rem 0;
               font-size: 0.98rem;}
      .banner-cache {background: #e8f1fb; color: #14538a;}
      .banner-save  {background: #e9f7ee; color: #1a7f3c;}
      .banner-warn  {background: #fdf6e3; color: #8a6d1a;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.markdown("## Settings")

    st.selectbox("OCR engine", ["PaddleOCR (PP-StructureV3)"], index=0)

    st.markdown("**Scope**")
    scope = st.radio(
        "Scope", ["Whole document (all pages)", "Page range"],
        label_visibility="collapsed",
    )
    page_range = None
    if scope == "Page range":
        c1, c2 = st.columns(2)
        start = c1.number_input("From", min_value=1, value=1, step=1)
        end = c2.number_input("To", min_value=1, value=5, step=1)
        page_range = (int(start), int(end))

    fast_mode = st.toggle(
        "Fast mode", value=True,
        help="Fast = mobile OCR models (quicker). Off = server-tier models "
             "(slower, more accurate on small/dense financial digits).",
    )
    tier = "fast" if fast_mode else "accurate"

    st.markdown("---")
    device = os.environ.get("OCR_DEVICE", "cpu")
    if device == "cpu":
        st.caption(
            "⏱ Scanned docs run OCR locally on CPU. The whole document can take "
            "several minutes on first run (models download once, then cached). "
            "Re-uploading the same PDF loads instantly from cache."
        )
    else:
        st.caption(
            f"⚡ Scanned docs run OCR on GPU ({device}). Models download once, "
            "then cached. Re-uploading the same PDF loads instantly from cache."
        )


# ---- header ----------------------------------------------------------------
st.markdown("# 📊 Financial Statement Extractor")
st.write(
    "Upload a financial PDF (scanned or digital). Extracts all tables and text, "
    "auto-detects **Balance Sheet / P&L / Cash Flow**, and reconciles the numbers."
)

uploaded = st.file_uploader("Upload a financial PDF", type=["pdf"])
run = st.button("Extract", type="primary", disabled=uploaded is None)


# ---- rendering helpers -----------------------------------------------------

def render_recon_banner(result):
    if result is None:
        return
    if result.passed:
        st.markdown(
            f'<div class="banner banner-save">✅ {result.detail}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="banner banner-warn">⚠️ {result.detail}</div>',
            unsafe_allow_html=True,
        )


def render_statement(statement):
    if statement.page_no is None:
        st.info(f"No {statement.name} detected in this document.")
        return
    st.caption(f"Page {statement.page_no}")
    render_recon_banner(statement.recon)
    if not statement.tables:
        st.warning("Statement page detected but no table parsed on it.")
    for t in statement.tables:
        if t.df is not None:
            st.dataframe(t.df, use_container_width=True)
        else:
            st.code(t.html[:2000], language="html")


# ---- main flow -------------------------------------------------------------

def run_extraction(pdf_path):
    prog = st.progress(0.0, text="Starting…")
    seen = {"n": 0}

    def cb(done, total, msg):
        seen["n"] = done
        prog.progress(min(done / 60.0, 0.95), text=f"{msg}…")

    extraction = ocr_engine.extract(
        pdf_path, tier=tier, device=device,
        page_range=page_range, progress=cb,
    )
    prog.progress(1.0, text="Done")
    prog.empty()
    return extraction


if run and uploaded is not None:
    # Persist upload under its REAL filename (stable stem => stable cache key;
    # ocr_engine additionally hashes the bytes so different files never collide).
    upload_dir = os.path.join(tempfile.gettempdir(), "fin_extractor_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    pdf_path = os.path.join(upload_dir, uploaded.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded.getbuffer())

    with st.spinner("Extracting…"):
        extraction = run_extraction(pdf_path)
        statements, all_tables = clf.classify(extraction)
        for s in statements.values():
            s.recon = rec.check(s)
        out_dir = exports.write_outputs(extraction, statements)

    st.session_state["result"] = {
        "extraction": extraction,
        "statements": statements,
        "all_tables": all_tables,
        "out_dir": out_dir,
    }


# ---- results ---------------------------------------------------------------
if "result" in st.session_state:
    r = st.session_state["result"]
    extraction = r["extraction"]
    statements = r["statements"]
    all_tables = r["all_tables"]

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Document type", extraction.doc_type)
    c2.metric("Engine", f"PaddleOCR · {extraction.tier}")
    c3.metric("Core statements", clf.core_statement_count(statements))
    c4.metric("Tables extracted", len(all_tables))

    if extraction.from_cache:
        st.markdown(
            '<div class="banner banner-cache">⚡ Loaded from cache — this '
            'document was extracted before (no re-OCR).</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="banner banner-save">💾 Auto-saved to '
        f'<code>{r["out_dir"]}</code> (Excel + JSON + Markdown)</div>',
        unsafe_allow_html=True,
    )

    tabs = st.tabs(
        ["Balance Sheet", "Profit & Loss", "Cash Flow",
         "📄 All tables", "📝 Full text"]
    )
    with tabs[0]:
        render_statement(statements["balance_sheet"])
    with tabs[1]:
        render_statement(statements["profit_loss"])
    with tabs[2]:
        render_statement(statements["cash_flow"])
    with tabs[3]:
        if not all_tables:
            st.info("No tables detected.")
        for t in all_tables:
            st.caption(f"Page {t.page_no}")
            if t.df is not None:
                st.dataframe(t.df, use_container_width=True)
            else:
                st.code(t.html[:2000], language="html")
    with tabs[4]:
        st.text_area("Full text", extraction.full_text, height=500)

    st.markdown("---")
    st.markdown("### Download")
    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "⬇️ Excel", exports.build_excel(extraction),
        file_name="document.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    d2.download_button(
        "⬇️ JSON", exports.build_json(extraction, statements),
        file_name="document.json", mime="application/json",
    )
    d3.download_button(
        "⬇️ Markdown", exports.build_markdown(extraction),
        file_name="document.md", mime="text/markdown",
    )
