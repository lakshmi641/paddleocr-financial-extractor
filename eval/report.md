# Extraction accuracy report

## Summary

- **Field-level accuracy: 9/9 (100.0%)** -- of the fields you supplied ground truth for, how many did the pipeline extract correctly (within 1% + 2 tolerance).
- Reconciliation pass rate: 3/3 (100.0%) -- statements whose accounting identity check passed (no ground truth needed, but can't catch errors that cancel out).

## Accuracy by field

| Field | Matched | Total | Accuracy |
|---|---|---|---|
| balance_sheet.total_assets | 1 | 1 | 100% |
| balance_sheet.total_equity | 1 | 1 | 100% |
| balance_sheet.total_liabilities | 1 | 1 | 100% |
| cash_flow.closing | 1 | 1 | 100% |
| cash_flow.net_change | 1 | 1 | 100% |
| cash_flow.opening | 1 | 1 | 100% |
| profit_loss.profit_before_tax | 1 | 1 | 100% |
| profit_loss.total_expenses | 1 | 1 | 100% |
| profit_loss.total_income | 1 | 1 | 100% |

## Per-document detail

### financial_note

- Source: `eval/pdfs/financial_note.pdf`

**balance_sheet** (page 12) ✅ Balanced: Total Assets (7,145) = Equity (5,178) + Liabilities (1,967).

| Field | Expected | Extracted | Match |
|---|---|---|---|
| total_assets | 7,145 | 7,145 | ✅ |
| total_liabilities | 1,967 | 1,967 | ✅ |
| total_equity | 5,178 | 5,178 | ✅ |

**profit_loss** (page 13) ✅ Ties out: Total Income (3,548) - Total Expenses (2,924) = Profit before tax (624).

| Field | Expected | Extracted | Match |
|---|---|---|---|
| total_income | 3,548 | 3,548 | ✅ |
| total_expenses | 2,924 | 2,924 | ✅ |
| profit_before_tax | 624 | 624 | ✅ |

**cash_flow** (page 14) ✅ Ties out: opening (396) + net change (-307) = closing (89).

| Field | Expected | Extracted | Match |
|---|---|---|---|
| opening | 396 | 396 | ✅ |
| closing | 89 | 89 | ✅ |
| net_change | -307 | -307 | ✅ |
