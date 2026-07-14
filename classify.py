"""
classify.py

Assign OCR'd pages/tables to the three core financial statements
(Balance Sheet, Profit & Loss, Cash Flow) using keyword scoring.

Pure functions over Extraction data — no Streamlit, no paddle. Unit-testable
against the cached output/page_*.json fixtures.
"""

import io
from dataclasses import dataclass, field

import pandas as pd

import table_repair

STATEMENTS = ["balance_sheet", "profit_loss", "cash_flow"]

STATEMENT_LABELS = {
    "balance_sheet": "Balance Sheet",
    "profit_loss": "Profit & Loss",
    "cash_flow": "Cash Flow",
}

KEYWORDS = {
    "balance_sheet": [
        "balance sheet", "equity and liabilities", "total assets",
        "total equity", "non-current assets", "current liabilities",
    ],
    "profit_loss": [
        "profit and loss", "statement of profit", "revenue from operations",
        "total income", "profit for the year", "earnings per",
    ],
    "cash_flow": [
        "cash flow", "operating activities", "investing activities",
        "financing activities", "cash and cash equivalents",
    ],
}


@dataclass
class TableRef:
    page_no: int
    order: int
    html: str
    df: pd.DataFrame = None


@dataclass
class Statement:
    key: str
    name: str
    page_no: int = None
    score: int = 0
    tables: list = field(default_factory=list)
    recon: object = None  # filled by reconcile.check


def _page_text(page):
    return " ".join(str(b.content) for b in page.blocks).lower()


def _score(text, key):
    return sum(text.count(w) for w in KEYWORDS[key])


def html_to_df(html):
    """Parse a table's HTML into a DataFrame with flat, unique string columns
    (so openpyxl / to_markdown don't choke on MultiIndex or duplicate names)."""
    try:
        dfs = pd.read_html(io.StringIO(html))
    except Exception:
        return None
    if not dfs:
        return None
    df = dfs[0]

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join(str(c) for c in col if c not in (None, "")).strip()
            for col in df.columns
        ]

    seen, new_cols = {}, []
    for col in df.columns:
        col = str(col)
        if col in seen:
            seen[col] += 1
            col = f"{col}_{seen[col]}"
        else:
            seen[col] = 0
        new_cols.append(col)
    df.columns = new_cols
    return df


def classify(extraction):
    """
    Returns (statements: dict[key->Statement], all_tables: list[TableRef]).

    Strategy: for each page compute per-statement keyword scores. A page is a
    candidate for a statement if that statement is its top score AND the page
    contains a table. The highest-scoring candidate page becomes THE page for
    that statement; its tables attach there. Every table is also collected into
    all_tables regardless.
    """
    all_tables = []
    page_scores = {}   # page_no -> {key: score}
    page_has_table = {}

    for pg in extraction.pages:
        text = _page_text(pg)
        scores = {k: _score(text, k) for k in STATEMENTS}
        page_scores[pg.page_no] = scores
        tables_here = [
            TableRef(
                pg.page_no, b.order, b.content,
                table_repair.table_df(b.content, b.table_words, html_to_df),
            )
            for b in pg.blocks
            if b.label == "table" and b.content
        ]
        page_has_table[pg.page_no] = bool(tables_here)
        all_tables.extend(tables_here)

    statements = {
        k: Statement(key=k, name=STATEMENT_LABELS[k]) for k in STATEMENTS
    }

    # Pick the best table-bearing page for each statement.
    for key in STATEMENTS:
        best_pg, best_score = None, 0
        for pg_no, scores in page_scores.items():
            s = scores[key]
            # must be this statement's territory and have a table
            if s > 0 and page_has_table.get(pg_no) and s == max(scores.values()):
                if s > best_score:
                    best_pg, best_score = pg_no, s
        if best_pg is not None:
            statements[key].page_no = best_pg
            statements[key].score = best_score
            statements[key].tables = [
                t for t in all_tables if t.page_no == best_pg
            ]

    return statements, all_tables


def core_statement_count(statements):
    return sum(1 for s in statements.values() if s.page_no is not None)
