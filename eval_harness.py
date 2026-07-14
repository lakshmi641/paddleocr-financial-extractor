"""
eval_harness.py

Measures real field-level accuracy of the extraction pipeline against a set
of manually-verified financial statements, instead of relying on eyeballing
one document.

Usage:
    1. Drop real financial statement PDFs into eval/pdfs/.
    2. For each one, add an entry to eval/ground_truth.json with the true
       values you read off the PDF yourself (latest year column only, same
       convention reconcile.py uses). Leave a field out if you're not sure
       of it -- an omitted field is skipped, not scored as wrong.
    3. Run:  python eval_harness.py
       (add --tier fast to force a tier; per-entry "tier" in the JSON wins)

Reports, per document and in aggregate:
    - field-level accuracy: of the fields you supplied ground truth for,
      what fraction did the pipeline extract correctly (within tolerance)?
    - reconciliation pass rate: what fraction of statements' accounting
      identity checks (reconcile.check) passed?

Neither number alone is "the" accuracy: field-level accuracy is the real
one but only as good as the ground truth you enter; reconciliation pass
rate is a free proxy that needs no ground truth but can't catch two errors
that cancel out.

Pure orchestration over ocr_engine/classify/reconcile -- no Streamlit.
"""

import argparse
import json
import os
import sys

import ocr_engine
import classify as clf
import reconcile as rec

DEFAULT_GROUND_TRUTH = os.path.join("eval", "ground_truth.json")
DEFAULT_REPORT = os.path.join("eval", "report.md")

# Same shape as reconcile.py's per-statement figure dicts.
STATEMENT_FIELDS = {
    "balance_sheet": ["total_assets", "total_liabilities", "total_equity"],
    "profit_loss": ["total_income", "total_expenses", "profit_before_tax"],
    "cash_flow": ["opening", "closing", "net_change"],
}


def values_match(extracted, expected, tol_pct=0.01, tol_abs=2):
    """Same tolerance policy as reconcile.py's identity checks: 1% of the
    larger magnitude plus 2 absolute (rounding in Lakhs)."""
    if extracted is None or expected is None:
        return False
    tol = max(abs(extracted), abs(expected)) * tol_pct + tol_abs
    return abs(extracted - expected) <= tol


def load_ground_truth(path):
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def resolve_pages(name, entry, default_tier, default_engine):
    """Prefer OCR-ing the PDF fresh (cache-aware); fall back to an
    already-extracted cache_dir so the harness can self-test against
    documents whose original PDF isn't on hand."""
    tier = entry.get("tier", default_tier)
    engine = entry.get("engine", default_engine)
    pdf_path = entry.get("pdf")
    if pdf_path and os.path.exists(pdf_path):
        extraction = ocr_engine.extract(pdf_path, tier=tier, engine=engine)
        return extraction.pages, tier, engine, pdf_path

    cache_dir = entry.get("cache_dir")
    if cache_dir and ocr_engine.has_cache(cache_dir):
        pages = ocr_engine.load_pages_from_dir(cache_dir)
        return pages, tier, engine, cache_dir

    return None, tier, engine, None


def evaluate_entry(name, entry, default_tier, default_engine):
    pages, tier, engine, source = resolve_pages(name, entry, default_tier, default_engine)
    if pages is None:
        return {
            "name": name, "source": None, "engine": engine, "error": (
                f"no readable 'pdf' ({entry.get('pdf')}) or cached "
                f"'cache_dir' ({entry.get('cache_dir')})"
            ),
            "statements": {},
        }

    extraction = ocr_engine.Extraction(pages=pages, tier=tier, engine=engine)
    statements, _ = clf.classify(extraction)

    result = {
        "name": name, "source": source, "engine": engine, "error": None,
        "statements": {},
    }
    for key, fields in STATEMENT_FIELDS.items():
        gt = entry.get(key)
        if not gt:
            continue
        statement = statements[key]
        recon = rec.check(statement)
        extracted = rec.extract_figures(statement)

        field_results = {}
        for field in fields:
            if field not in gt:
                continue
            field_results[field] = {
                "expected": gt[field],
                "extracted": extracted.get(field),
                "match": values_match(extracted.get(field), gt[field]),
            }

        result["statements"][key] = {
            "detected": statement.page_no is not None,
            "page_no": statement.page_no,
            "recon_status": recon.status,
            "recon_detail": recon.detail,
            "fields": field_results,
        }
    return result


def summarize(results):
    total_fields = matched_fields = 0
    total_statements = passed_recon = 0
    per_field_type = {}  # "balance_sheet.total_assets" -> [n, matched]
    per_engine = {}  # "paddleocr" -> {"fields": [n, matched], "recon": [n, matched]}

    for r in results:
        engine = r.get("engine") or "paddleocr"
        eslot = per_engine.setdefault(engine, {"fields": [0, 0], "recon": [0, 0]})
        for key, s in r["statements"].items():
            total_statements += 1
            eslot["recon"][0] += 1
            if s["recon_status"] == rec.OK:
                passed_recon += 1
                eslot["recon"][1] += 1
            for field, f in s["fields"].items():
                total_fields += 1
                slot = per_field_type.setdefault(f"{key}.{field}", [0, 0])
                slot[0] += 1
                eslot["fields"][0] += 1
                if f["match"]:
                    matched_fields += 1
                    slot[1] += 1
                    eslot["fields"][1] += 1

    return {
        "total_fields": total_fields,
        "matched_fields": matched_fields,
        "field_accuracy": (matched_fields / total_fields) if total_fields else None,
        "total_statements": total_statements,
        "passed_recon": passed_recon,
        "recon_rate": (passed_recon / total_statements) if total_statements else None,
        "per_field_type": per_field_type,
        "per_engine": per_engine,
    }


def render_report(results, summary):
    lines = ["# Extraction accuracy report\n"]

    lines.append("## Summary\n")
    if summary["total_fields"]:
        lines.append(
            f"- **Field-level accuracy: {summary['matched_fields']}/"
            f"{summary['total_fields']} "
            f"({summary['field_accuracy']*100:.1f}%)** -- of the fields you "
            "supplied ground truth for, how many did the pipeline extract "
            "correctly (within 1% + 2 tolerance)."
        )
    else:
        lines.append("- No ground-truth fields were supplied to score.")
    if summary["total_statements"]:
        lines.append(
            f"- Reconciliation pass rate: {summary['passed_recon']}/"
            f"{summary['total_statements']} "
            f"({summary['recon_rate']*100:.1f}%) -- statements whose "
            "accounting identity check passed (no ground truth needed, "
            "but can't catch errors that cancel out)."
        )
    lines.append("")

    if len(summary["per_engine"]) > 1:
        lines.append("## Accuracy by engine\n")
        lines.append("| Engine | Field accuracy | Reconciliation pass rate |")
        lines.append("|---|---|---|")
        for engine, e in sorted(summary["per_engine"].items()):
            fn, fm = e["fields"]
            rn, rm = e["recon"]
            f_pct = f"{fm}/{fn} ({fm/fn*100:.0f}%)" if fn else "—"
            r_pct = f"{rm}/{rn} ({rm/rn*100:.0f}%)" if rn else "—"
            lines.append(f"| {engine} | {f_pct} | {r_pct} |")
        lines.append("")

    if summary["per_field_type"]:
        lines.append("## Accuracy by field\n")
        lines.append("| Field | Matched | Total | Accuracy |")
        lines.append("|---|---|---|---|")
        for field, (n, matched) in sorted(summary["per_field_type"].items()):
            lines.append(f"| {field} | {matched} | {n} | {matched/n*100:.0f}% |")
        lines.append("")

    lines.append("## Per-document detail\n")
    for r in results:
        lines.append(f"### {r['name']}\n")
        if r["error"]:
            lines.append(f"- ⚠️ Skipped: {r['error']}\n")
            continue
        lines.append(f"- Source: `{r['source']}` (engine: `{r['engine']}`)\n")
        if not r["statements"]:
            lines.append("- No ground truth fields supplied for this document.\n")
            continue
        for key, s in r["statements"].items():
            status = "✅" if s["recon_status"] == rec.OK else "⚠️"
            lines.append(
                f"**{key}** (page {s['page_no']}) {status} {s['recon_detail']}\n"
            )
            if s["fields"]:
                lines.append("| Field | Expected | Extracted | Match |")
                lines.append("|---|---|---|---|")
                for field, f in s["fields"].items():
                    mark = "✅" if f["match"] else "❌"
                    extracted_str = (
                        "—" if f["extracted"] is None else f"{f['extracted']:,.0f}"
                    )
                    lines.append(
                        f"| {field} | {f['expected']:,} | {extracted_str} | {mark} |"
                    )
                lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--tier", default="accurate", choices=["fast", "accurate"])
    parser.add_argument(
        "--engine", default="paddleocr", choices=["paddleocr", "paddleocr_vl"],
        help="Default engine for entries that don't set their own 'engine' "
             "field. Add two entries for the same PDF (one per engine, "
             "different names) to A/B them in one report.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.ground_truth):
        print(f"No ground truth file at {args.ground_truth}", file=sys.stderr)
        sys.exit(1)

    entries = load_ground_truth(args.ground_truth)
    if not entries:
        print("Ground truth file has no document entries yet.", file=sys.stderr)
        sys.exit(1)

    results = []
    for name, entry in entries.items():
        print(f"Evaluating {name}...")
        results.append(evaluate_entry(name, entry, args.tier, args.engine))

    summary = summarize(results)
    report = render_report(results, summary)

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf8") as f:
        f.write(report)

    print()
    if summary["total_fields"]:
        print(
            f"Field-level accuracy: {summary['matched_fields']}/"
            f"{summary['total_fields']} ({summary['field_accuracy']*100:.1f}%)"
        )
    if summary["total_statements"]:
        print(
            f"Reconciliation pass rate: {summary['passed_recon']}/"
            f"{summary['total_statements']} ({summary['recon_rate']*100:.1f}%)"
        )
    print(f"Full report written to {args.report}")


if __name__ == "__main__":
    main()
