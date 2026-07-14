"""
reconcile.py

Best-effort numeric sanity checks on classified statements. REVIEWER AID, not a
guarantee — OCR'd scanned tables (merged cells, note-number leakage, Rs. Lakhs)
parse imperfectly, so anything we cannot confidently parse/verify returns
"could_not_verify" rather than a false pass.

Number parsing here is built against the real OCR output of scanned Indian
financial statements, which have three recurring quirks:
  1. Thousands separators: "1,967" -> 1967 (comma is NOT a value delimiter).
  2. Two year-columns merged into one cell: "1,967 1,706" -> [1967, 1706]
     (whitespace IS a value delimiter). We take the first = latest year.
  3. Parenthesised negatives: "(307)" -> -307.

Pure functions — no Streamlit, no paddle.
"""

import re
from dataclasses import dataclass

import pandas as pd

OK = "ok"
COULD_NOT_VERIFY = "could_not_verify"

# tolerance: 1% of the larger magnitude, plus 2 absolute (rounding in Lakhs)
def _tol(a, b):
    return max(abs(a), abs(b)) * 0.01 + 2


@dataclass
class ReconResult:
    status: str = COULD_NOT_VERIFY
    detail: str = "Could not verify (totals not found / parsed). Please review."

    @property
    def passed(self):
        return self.status == OK


def _numbers_in(cell):
    """Parse every number in one cell. Splits on whitespace (merged year
    columns), keeps commas as thousands separators, treats (x) as negative."""
    if cell is None:
        return []
    s = str(cell).strip()
    if not s or s.lower() == "nan":
        return []
    out = []
    for tok in s.split():
        neg = "(" in tok and ")" in tok
        cleaned = re.sub(r"[^0-9.]", "", tok.replace(",", ""))
        if cleaned in ("", ".", "-"):
            continue
        try:
            num = float(cleaned)
        except ValueError:
            continue
        out.append(-num if neg else num)
    return out


def _norm(text):
    """Lowercase and remove all whitespace, so OCR spacing noise
    ('TOTALASSETS', 'Total  assets') collapses to a stable key."""
    return re.sub(r"\s+", "", str(text)).lower()


def _find_row_numbers(df, include, exclude=()):
    """First row whose normalized full text matches an `include` key, no
    `exclude` key, AND actually carries numbers. Skips label-only rows (OCR
    sometimes dumps the whole label column into one number-less cell). Returns
    the row's value numbers, or None."""
    if df is None or df.empty:
        return None
    for _, row in df.iterrows():
        cells = [str(c) for c in row.tolist()]
        key = _norm(" ".join(cells))
        if not any(x in key for x in include):
            continue
        if any(x in key for x in exclude):
            continue
        nums = []
        for c in cells[1:]:
            nums.extend(_numbers_in(c))
        if nums:
            return nums
    return None


def _first(nums):
    return nums[0] if nums else None


def _first_df(statement):
    for t in statement.tables:
        if isinstance(t.df, pd.DataFrame) and not t.df.empty:
            return t.df
    return None


# ---- Balance Sheet ---------------------------------------------------------
# Identity: Total Assets == Total Equity + Total Liabilities (latest year).

def extract_balance_sheet_figures(df):
    """Best-effort latest-year figures: {total_assets, total_liabilities,
    total_equity}. Any value is None if it couldn't be found/parsed. Exposed
    separately from check_balance_sheet so callers (e.g. the eval harness)
    can compare the raw numbers against ground truth, not just pass/fail."""
    if df is None:
        return {"total_assets": None, "total_liabilities": None, "total_equity": None}

    assets = _first(_find_row_numbers(df, ["totalassets"]))
    liabilities = _first(
        _find_row_numbers(
            df, ["totalliabilities"],
            exclude=["totalcurrentliabilities", "totalnon-currentliabilities",
                     "totalnoncurrentliabilities"],
        )
    )
    # Equity total: largest equity-labelled value that isn't a liabilities/
    # share-capital sub-line and isn't larger than total assets.
    equity = None
    if assets:
        best = 0.0
        for _, row in df.iterrows():
            cells = [str(c) for c in row.tolist()]
            key = _norm(" ".join(cells))
            if "equity" not in key:
                continue
            if any(x in key for x in ("liabilit", "sharecapital")):
                continue
            for c in cells[1:]:
                for n in _numbers_in(c):
                    if 0 < n <= assets + _tol(assets, assets) and n > best:
                        best = n
        equity = best or None

    return {"total_assets": assets, "total_liabilities": liabilities, "total_equity": equity}


def check_balance_sheet(statement):
    df = _first_df(statement)
    if df is None:
        return ReconResult()

    figs = extract_balance_sheet_figures(df)
    assets = figs["total_assets"]
    liabilities = figs["total_liabilities"]
    equity = figs["total_equity"]

    if assets is None or liabilities is None or equity is None:
        return ReconResult()

    if abs(assets - (equity + liabilities)) <= _tol(assets, equity + liabilities):
        return ReconResult(
            OK,
            f"Balanced: Total Assets ({assets:,.0f}) = Equity ({equity:,.0f}) "
            f"+ Liabilities ({liabilities:,.0f}).",
        )
    return ReconResult(
        COULD_NOT_VERIFY,
        f"Mismatch: Assets {assets:,.0f} vs Equity+Liabilities "
        f"{equity + liabilities:,.0f} ({equity:,.0f}+{liabilities:,.0f}). "
        "Please review.",
    )


# ---- Profit & Loss ----------------------------------------------------------
# Identity: Total Income - Total Expenses == Profit before tax (latest year).
# (Deliberately not Profit before tax - Tax expense == Profit for the period:
# the tax-expense rows are usually 2-3 sub-lines with a much higher OCR error
# rate than a single "Total Income"/"Total expenses"/"Profit before tax" row.)

def extract_profit_loss_figures(df):
    """Best-effort latest-year figures: {total_income, total_expenses,
    profit_before_tax}. Any value is None if it couldn't be found/parsed."""
    if df is None:
        return {"total_income": None, "total_expenses": None, "profit_before_tax": None}
    return {
        "total_income": _first(_find_row_numbers(df, ["totalincome"])),
        "total_expenses": _first(_find_row_numbers(df, ["totalexpenses"])),
        "profit_before_tax": _first(_find_row_numbers(df, ["profitbeforetax"])),
    }


def check_profit_loss(statement):
    df = _first_df(statement)
    if df is None:
        return ReconResult()

    figs = extract_profit_loss_figures(df)
    income = figs["total_income"]
    expenses = figs["total_expenses"]
    pbt = figs["profit_before_tax"]

    if income is None or expenses is None or pbt is None:
        return ReconResult(
            COULD_NOT_VERIFY,
            "Could not verify (Total Income / Total Expenses / Profit "
            "before tax not found or parsed). Please review.",
        )

    if abs((income - expenses) - pbt) <= _tol(pbt, income - expenses):
        return ReconResult(
            OK,
            f"Ties out: Total Income ({income:,.0f}) - Total Expenses "
            f"({expenses:,.0f}) = Profit before tax ({pbt:,.0f}).",
        )
    return ReconResult(
        COULD_NOT_VERIFY,
        f"Mismatch: Income - Expenses ({income - expenses:,.0f}) != "
        f"Profit before tax ({pbt:,.0f}). Please review.",
    )


# ---- Cash Flow -------------------------------------------------------------
# Identity: opening + net change == closing (latest year).

def extract_cash_flow_figures(df):
    """Best-effort latest-year figures: {opening, closing, net_change}. Any
    value is None if it couldn't be found/parsed."""
    if df is None:
        return {"opening": None, "closing": None, "net_change": None}
    return {
        "opening": _first(_find_row_numbers(df, ["atthebeginningof"])),
        "closing": _first(_find_row_numbers(df, ["attheendof", "attheyearend"])),
        "net_change": _first(
            _find_row_numbers(
                df, ["netincrease", "netdecrease", "net(decrease)",
                     "net(increase)", "netchangeincash"],
            )
        ),
    }


def check_cash_flow(statement):
    df = _first_df(statement)
    if df is None:
        return ReconResult()

    figs = extract_cash_flow_figures(df)
    opening = figs["opening"]
    closing = figs["closing"]
    net = figs["net_change"]

    if opening is None or closing is None or net is None:
        return ReconResult()

    if abs((opening + net) - closing) <= _tol(closing, opening + net):
        return ReconResult(
            OK,
            f"Ties out: opening ({opening:,.0f}) + net change ({net:,.0f}) "
            f"= closing ({closing:,.0f}).",
        )
    return ReconResult(
        COULD_NOT_VERIFY,
        f"Opening+net ({opening + net:,.0f}) != closing ({closing:,.0f}). "
        "Please review.",
    )


def check(statement):
    if statement.page_no is None:
        return ReconResult(COULD_NOT_VERIFY, "Statement not detected.")
    if statement.key == "balance_sheet":
        return check_balance_sheet(statement)
    if statement.key == "cash_flow":
        return check_cash_flow(statement)
    return check_profit_loss(statement)


_FIGURE_EXTRACTORS = {
    "balance_sheet": extract_balance_sheet_figures,
    "profit_loss": extract_profit_loss_figures,
    "cash_flow": extract_cash_flow_figures,
}


def extract_figures(statement):
    """Best-effort latest-year figures for any statement, keyed the same way
    regardless of statement type. Used by the eval harness to score field-
    level accuracy against ground truth (check() only reports pass/fail)."""
    extractor = _FIGURE_EXTRACTORS.get(statement.key)
    if extractor is None or statement.page_no is None:
        return {}
    return extractor(_first_df(statement))
