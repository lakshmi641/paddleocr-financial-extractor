import os
import json
import time
import inspect
import pandas as pd

from paddleocr import PPStructureV3

INPUT_PDF = "input/financial_note.pdf"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def env_bool(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def env_int(name, default):
    val = os.environ.get(name)
    return int(val) if val else default


# All of these are overridable via env vars so the same script runs
# unchanged on the CPU dev machine and the GPU test VM.
#
# Param names below match PaddleOCR 3.x PPStructureV3 (verified against the
# 3.x constructor signature). Notably: the table structure model is split into
# a WIRED and a WIRELESS model — there is no single "table_recognition_model_name".
# device / precision / cpu_threads are consumed via **kwargs by the pipeline base,
# so they don't appear as named params but ARE valid.
DEVICE = os.environ.get("OCR_DEVICE", "cpu")               # e.g. "gpu:0" on the VM
PRECISION = os.environ.get("OCR_PRECISION")                 # e.g. "fp16" on GPU
WIRED_TABLE_MODEL = os.environ.get("OCR_WIRED_TABLE_MODEL", "SLANet_plus")
WIRELESS_TABLE_MODEL = os.environ.get("OCR_WIRELESS_TABLE_MODEL")
TEXT_DET_MODEL = os.environ.get("OCR_TEXT_DET_MODEL")        # e.g. PP-OCRv5_server_det
TEXT_REC_MODEL = os.environ.get("OCR_TEXT_REC_MODEL")        # e.g. PP-OCRv5_server_rec
TEXT_DET_LIMIT_SIDE_LEN = env_int("OCR_TEXT_DET_LIMIT_SIDE_LEN", 4000)
TEXT_DET_LIMIT_TYPE = os.environ.get("OCR_TEXT_DET_LIMIT_TYPE", "max")
USE_DOC_ORIENTATION = env_bool("OCR_USE_DOC_ORIENTATION", False)
USE_DOC_UNWARPING = env_bool("OCR_USE_DOC_UNWARPING", False)
USE_TEXTLINE_ORIENTATION = env_bool("OCR_USE_TEXTLINE_ORIENTATION", False)

candidate_kwargs = {
    "device": DEVICE,
    "use_formula_recognition": False,
    "use_seal_recognition": False,
    "use_chart_recognition": False,
    "use_doc_orientation_classify": USE_DOC_ORIENTATION,
    "use_doc_unwarping": USE_DOC_UNWARPING,
    "use_textline_orientation": USE_TEXTLINE_ORIENTATION,
    "wired_table_structure_recognition_model_name": WIRED_TABLE_MODEL,
    "wireless_table_structure_recognition_model_name": WIRELESS_TABLE_MODEL,
    "text_det_limit_side_len": TEXT_DET_LIMIT_SIDE_LEN,
    "text_det_limit_type": TEXT_DET_LIMIT_TYPE,
    "precision": PRECISION,
    "text_detection_model_name": TEXT_DET_MODEL,
    "text_recognition_model_name": TEXT_REC_MODEL,
}
if DEVICE == "cpu":
    candidate_kwargs["cpu_threads"] = env_int("OCR_CPU_THREADS", 8)

# PPStructureV3.__init__ ends in **kwargs: many valid options (device, precision,
# cpu_threads, ...) are absorbed there and are NOT explicit named params. So only
# filter by the named-param whitelist when the signature has NO **kwargs; if it
# does, pass everything through (the pipeline validates the rest itself).
sig_params = inspect.signature(PPStructureV3.__init__).parameters
has_var_kwargs = any(
    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values()
)
named_params = set(sig_params)

init_kwargs = {}
skipped = {}
for key, value in candidate_kwargs.items():
    if value is None:
        continue
    if has_var_kwargs or key in named_params:
        init_kwargs[key] = value
    else:
        skipped[key] = value

print("=" * 70)
print("Initializing PPStructureV3...")
print("=" * 70)
print("Using kwargs :", init_kwargs)
if skipped:
    print("Skipped (not supported by this paddleocr version):", skipped)

pipeline = PPStructureV3(**init_kwargs)

print("Pipeline initialized")
print()

text_rows = []
tables = []
markdown_pages = []

start = time.time()

page_count = 0

for page in pipeline.predict_iter(INPUT_PDF):

    page_count += 1

    print(f"\nProcessing Page {page_count}")

    page_start = time.time()

    page.save_to_json(
        os.path.join(
            OUTPUT_DIR,
            f"page_{page_count}.json"
        )
    )

    if hasattr(page, "markdown"):
        markdown_pages.append(page.markdown)

    if hasattr(page, "res"):

        for block in page.res:

            block_type = block.get("type", "")

            if block_type == "text":

                text_rows.append(
                    {
                        "page": page_count,
                        "text": block.get("text", "")
                    }
                )

            elif block_type == "table":

                html = block.get("res", "")

                try:

                    dfs = pd.read_html(html)

                    for df in dfs:

                        df.insert(
                            0,
                            "page",
                            page_count
                        )

                        tables.append(df)

                except Exception as e:

                    print("Table parse failed:", e)

    print(
        f"Finished Page {page_count} "
        f"({time.time()-page_start:.2f} sec)"
    )

print("\nSaving outputs...")

text_df = pd.DataFrame(text_rows)

text_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "text.csv"
    ),
    index=False
)

with pd.ExcelWriter(
    os.path.join(
        OUTPUT_DIR,
        "tables.xlsx"
    )
) as writer:

    if tables:

        for i, table in enumerate(tables):

            table.to_excel(
                writer,
                sheet_name=f"Table_{i+1}",
                index=False
            )

    else:

        pd.DataFrame(
            {"Info": ["No tables detected"]}
        ).to_excel(
            writer,
            index=False
        )

json_data = {
    "text": text_rows,
    "tables": [
        t.to_dict(orient="records")
        for t in tables
    ]
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "document.json"
    ),
    "w",
    encoding="utf8"
) as f:

    json.dump(
        json_data,
        f,
        indent=4,
        ensure_ascii=False
    )

if markdown_pages:

    markdown = pipeline.concatenate_markdown_pages(
        markdown_pages
    )

    with open(
        os.path.join(
            OUTPUT_DIR,
            "document.md"
        ),
        "w",
        encoding="utf8"
    ) as f:

        f.write(markdown["markdown_texts"])

print()
print("=" * 70)
print("Completed")
print("=" * 70)
total_time = time.time() - start
print(f"Device : {DEVICE}")
print(f"Pages : {page_count}")
print(f"Paragraphs : {len(text_rows)}")
print(f"Tables : {len(tables)}")
print(f"Total Time : {total_time:.2f} sec")
if page_count:
    print(f"Avg sec/page : {total_time/page_count:.2f}")