"""
table_repair.py

Fallback table reconstruction for when PPStructureV3's table *structure*
model mis-predicts the cell grid. On dense financial tables (40+ line items)
it occasionally collapses most of a column into one giant merged cell,
scrambling the HTML it hands back (a whole label column dumped into one
cell, numbers shifted into the wrong row). The underlying word-level OCR
(table_res_list[*].table_ocr_pred) is unaffected by that failure, so we
rebuild a plain row/column grid directly from those word boxes instead of
trusting the model's cell layout.

Pure functions — no Streamlit, no paddle.
"""

import re

import pandas as pd

# A cell this long only happens when the structure model merged many real
# rows into one cell (e.g. a whole label column dumped into row 0).
_BROKEN_CELL_LEN = 150

# A cell counts as "numeric" only if it IS a number (with thousands commas,
# a decimal point, or parenthesised-negative) end to end -- not merely
# "contains a digit somewhere", which would also match date text like
# "March 31,2025" in a header row and throw off row/column classification.
_NUMERIC_RE = re.compile(r"^\(?-?\d[\d,]*(\.\d+)?\)?$")


def is_broken_table(df):
    """Heuristic: any single cell holding a suspiciously long string means
    the table's cell grid is scrambled, not just noisy OCR."""
    if df is None or df.empty:
        return True
    for col in df.columns:
        for val in df[col].astype(str):
            if len(val) > _BROKEN_CELL_LEN:
                return True
    return False


def _looks_numeric(text):
    return bool(_NUMERIC_RE.match(text.strip()))


def rebuild_table_from_words(words):
    """Rebuild a table DataFrame from (text, (x0, y0, x1, y1)) word boxes.

    1. Cluster words into rows by y-center (tolerant of a bit of slope/noise
       within one printed line).
    2. Cluster words into columns by x0 (gap-based split).
    3. Merge the leading columns that never carry a numeric value anywhere
       in a data row into one "Description" column, and drop columns that
       are blank in every data row (header-only artifacts).

    Returns None if there isn't enough to build a sensible grid.
    """
    if not words:
        return None

    items = [
        {
            "text": text,
            "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3],
            "ycenter": (box[1] + box[3]) / 2,
            "height": box[3] - box[1],
        }
        for text, box in words
    ]
    if not items:
        return None

    # --- row clustering -----------------------------------------------
    items.sort(key=lambda w: w["ycenter"])
    heights = sorted(w["height"] for w in items)
    median_h = heights[len(heights) // 2] or 20
    row_tol = median_h * 0.7

    rows = []
    current = [items[0]]
    row_y = items[0]["ycenter"]
    for w in items[1:]:
        if abs(w["ycenter"] - row_y) <= row_tol:
            current.append(w)
            row_y = sum(x["ycenter"] for x in current) / len(current)
        else:
            rows.append(current)
            current = [w]
            row_y = w["ycenter"]
    rows.append(current)

    # --- column clustering ---------------------------------------------
    xs = sorted(w["x0"] for w in items)
    col_bounds = [xs[0]]
    for x in xs[1:]:
        if x - col_bounds[-1] > 80:
            col_bounds.append(x)

    def col_index(x0):
        return min(range(len(col_bounds)), key=lambda i: abs(x0 - col_bounds[i]))

    grid = []
    for row in rows:
        row.sort(key=lambda w: w["x0"])
        cells = [""] * len(col_bounds)
        for w in row:
            ci = col_index(w["x0"])
            cells[ci] = (cells[ci] + " " + w["text"]).strip()
        grid.append(cells)

    if len(col_bounds) < 2:
        return None

    # --- identify header rows (no standalone numeric value at all) vs data rows.
    # A financial statement's column header is 1-2 printed lines ("Notes" /
    # "As at" / "As at" then "March 31,2025" / "March 31,2024", sometimes
    # split across two OCR lines a few px apart). Cap the header search at 2
    # rows so that later no-number rows -- section labels like "ASSETS" or
    # "Non-current assets" that are real body rows, just with blank values --
    # aren't mistaken for more header lines and swallowed into a column name.
    _MAX_HEADER_ROWS = 2

    def row_has_number(cells):
        return any(_looks_numeric(c) for c in cells)

    first_value_idx = next(
        (i for i, r in enumerate(grid) if row_has_number(r)), len(grid)
    )
    header_cutoff = min(first_value_idx, _MAX_HEADER_ROWS)
    header_rows = grid[:header_cutoff]
    data_rows = grid[header_cutoff:] or grid

    # --- merge leading non-numeric columns into one Description column --
    value_col = next(
        (
            ci for ci in range(len(col_bounds))
            if any(_looks_numeric(r[ci]) for r in data_rows)
        ),
        1,
    )
    label_span = max(value_col, 1)

    # --- decide which columns beyond the description actually carry data --
    # A column earns its keep if some data row has a number in it. Columns
    # that never do (only ever populated on a header line, e.g. a stray
    # "March 31,2025" that landed one column over from "As at") get folded
    # into the nearest surviving column instead of silently dropped, so
    # header text isn't lost.
    value_cols = list(range(label_span, len(col_bounds)))
    keep_cols = [
        ci for ci in value_cols
        if any(_looks_numeric(r[ci]) for r in data_rows)
    ] or value_cols

    def nearest_keep(ci):
        return min(keep_cols, key=lambda k: abs(col_bounds[k] - col_bounds[ci]))

    remap = {ci: (ci if ci in keep_cols else nearest_keep(ci)) for ci in value_cols}

    def fold_row(row):
        desc = " ".join(c for c in row[:label_span] if c).strip()
        bucket = {ci: [] for ci in keep_cols}
        for ci in value_cols:
            if row[ci]:
                bucket[remap[ci]].append(row[ci])
        return [desc] + [" ".join(bucket[ci]).strip() for ci in keep_cols]

    header = [fold_row(row) for row in header_rows]
    body = [fold_row(row) for row in data_rows]

    # --- compound column names from the folded header rows, generic fallback --
    n_cols = 1 + len(keep_cols)
    names = []
    for c in range(n_cols):
        parts = [h[c] for h in header if h[c]]
        names.append(" ".join(dict.fromkeys(parts)) if parts else None)
    names[0] = names[0] or "Description"
    seen = {}
    columns = []
    for i, name in enumerate(names):
        name = name or f"Value {i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        columns.append(name)

    df = pd.DataFrame(body, columns=columns)
    return df


def table_df(html, table_words, html_to_df):
    """Best-effort DataFrame for one table: parse the model's HTML, but fall
    back to rebuilding from raw OCR word boxes if that HTML looks scrambled
    and word boxes are available. html_to_df is injected (classify.py's
    HTML parser) to avoid a circular import."""
    df = html_to_df(html)
    if table_words and is_broken_table(df):
        rebuilt = rebuild_table_from_words(table_words)
        if rebuilt is not None and not rebuilt.empty:
            return rebuilt
    return df
