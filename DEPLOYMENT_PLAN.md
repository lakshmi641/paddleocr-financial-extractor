# PaddleOCR GPU VM — Deployment & Re-Test Plan

Goal: re-run the financial-document OCR pipeline (`extract.py`) on a GCP GPU VM to
fix the two complaints — **speed** and **accuracy** — and measure the difference
against the current CPU baseline.

> **Key framing (read this first):** GPU does not make OCR more accurate by itself.
> What it does is make the **server-tier models affordable**. On your Mac (CPU) the
> pipeline defaults to *mobile* models to stay tolerable on speed — mobile models are
> the accuracy compromise. On a T4 GPU you can run the **server** detection/recognition
> models AND the better table model at high speed, so you get **both** at once. That is
> the whole thesis of this test.
>
> The source PDF is a **48-page image-only scan** (no text layer), ~1654×2338 px per
> page ≈ **200 DPI**. That 200 DPI is a hard ceiling — no model or GPU recovers detail
> the scanner never captured. Everything below works *within* that ceiling.

---

## 0. What will / won't improve

| Lever | Speed | Accuracy | Notes |
|---|---|---|---|
| `device=gpu:0` | ✅ big (5–10×) | — | The core speed fix |
| Batching / fp16 | ✅ | — | fp16 near-free on T4 |
| Server det/rec models (`PP-OCRv5_server_*`) | ⬇ slower than mobile, but fine on GPU | ✅ big for small/dense digits | The core accuracy fix |
| Table model `SLANet_plus` | slight ⬇ | ✅ merged-cell financial grids | |
| doc orientation + unwarping | slight ⬇ | ✅ skewed scanner pages | Cheap on GPU |
| `text_det_limit_side_len=4000, max` | — | ✅ stops pre-detection downsampling of tiny digits | |
| Higher DPI re-render / super-res | ⬇⬇ | ❌ speculative | **Skip** — source is capped ~200 DPI |

---

## 1. Pre-flight (run locally, no cost)

```bash
# 1a. Re-authenticate (your token expired)
gcloud auth login

# 1b. Confirm project
gcloud config set project julley-pms-dev
gcloud config get-value project

# 1c. Enable Compute Engine API (no-op if already on)
gcloud services enable compute.googleapis.com

# 1d. CHECK GPU QUOTA — fresh projects often have 0. This is the #1 blocker.
gcloud compute regions describe us-central1 \
  --format="table(quotas.metric,quotas.limit,quotas.usage)" \
  | grep -iE "NVIDIA_T4|PREEMPTIBLE_NVIDIA_T4|GPUS_ALL"
```

- If `NVIDIA_T4_GPUS` (and/or `PREEMPTIBLE_NVIDIA_T4_GPUS` for spot) limit is **0**,
  request an increase: GCP Console → IAM & Admin → **Quotas** → filter "T4" →
  region `us-central1` → request **1**. Approval is usually minutes–hours.
- **Cost saver:** we use a **Spot** VM, so the *preemptible* T4 quota is what matters.

```bash
# 1e. Confirm the Deep Learning VM image family (bundles CUDA + NVIDIA driver)
gcloud compute images list --project deeplearning-platform-release \
  --filter="family~common-cu12" --format="value(family)" | sort -u
```
- **Verified against GCP docs (July 2026):** use **`common-cu129-ubuntu-2204-nvidia-580`**
  — Ubuntu 22.04, ships **NVIDIA driver 580**. A 580 driver supports CUDA up to 12.9,
  so it runs *any* PaddlePaddle CUDA wheel (both the cu118 and cu126 wheels), which is
  exactly why we don't have to match the image's system CUDA to Paddle's.
- Only cu129 and cu128 offer Ubuntu 22.04; older CUDA versions are Debian-11 only.
  The command above just double-checks the family is still published in your region.

---

## 2. Create the VM (⚠️ first billable step)

```bash
export ZONE=us-central1-a
export VM=paddleocr-test
export IMAGE_FAMILY=common-cu129-ubuntu-2204-nvidia-580   # verified in 1e

gcloud compute instances create $VM \
  --zone=$ZONE \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=$IMAGE_FAMILY \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-ssd \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"
```

- `SPOT` → ~1/3 the price; fine for a throwaway test (can be preempted, just restart).
- `install-nvidia-driver=True` → DLVM auto-installs the matching NVIDIA driver on first boot.
- First boot runs the driver install; give it ~2–3 min before SSH-ing.

```bash
# SSH in (retry for a minute or two while it finishes first-boot setup)
gcloud compute ssh $VM --zone=$ZONE
```

---

## 3. Environment setup (on the VM)

```bash
# 3a. Verify the GPU + driver are live
nvidia-smi          # must show a Tesla T4 and a driver version. If not, wait & retry.

# 3b. Fresh venv (DLVM ships conda/py3.10; a venv keeps this isolated)
python3 -m venv ~/ocrenv
source ~/ocrenv/bin/activate
python -m pip install --upgrade pip

# 3c. Install PaddlePaddle GPU.
#     We use the CUDA 11.8 wheel ON PURPOSE: it bundles its own CUDA runtime and only
#     needs a driver >= 520 (any modern DLVM has this), so it avoids CUDA-minor-version
#     mismatch headaches regardless of the image's system CUDA.
python -m pip install paddlepaddle-gpu==3.0.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# 3d. Sanity check GPU is visible to Paddle
python -c "import paddle; print('paddle', paddle.__version__); print('compiled_with_cuda', paddle.is_compiled_with_cuda()); print('gpus', paddle.device.cuda.device_count())"
# Expect: compiled_with_cuda True, gpus 1

# 3e. Install PaddleOCR + the pipeline deps
python -m pip install "paddleocr>=3.0.0" pandas openpyxl lxml beautifulsoup4 tqdm
```

> If 3c's cu118 wheel ever fails to find the GPU in 3d, the fallback is the CUDA 12.6
> wheel: same command with `/cu126/`. Try cu118 first — it's the most compatible.

---

## 4. Transfer the script + input (from your Mac)

```bash
# From /Users/lakshmi/Downloads/ocr 2 on the Mac:
gcloud compute scp extract.py $VM:~/ --zone=$ZONE
gcloud compute scp --recurse input $VM:~/ --zone=$ZONE
gcloud compute ssh $VM --zone=$ZONE --command="mkdir -p ~/output"
```

`extract.py` is already env-var driven and filters out any kwarg the installed
paddleocr version doesn't support (it prints "Using kwargs" / "Skipped" at startup so
you can see exactly which optimizations took effect).

---

## 5. Run configurations (on the VM)

Run these **one at a time** so you know which lever did what. Each writes into
`~/output` (rename between runs to keep both sets).

```bash
cd ~
source ~/ocrenv/bin/activate

# --- Run A: GPU baseline (speed test, default models) ---
OCR_DEVICE=gpu:0 OCR_PRECISION=fp16 \
  python extract.py
mv output output_A_gpu_baseline && mkdir output

# --- Run B: GPU + accuracy tuning (server models, wired table model, deskew) ---
OCR_DEVICE=gpu:0 OCR_PRECISION=fp16 \
OCR_TEXT_DET_MODEL=PP-OCRv5_server_det \
OCR_TEXT_REC_MODEL=PP-OCRv5_server_rec \
OCR_WIRED_TABLE_MODEL=SLANet_plus \
OCR_USE_DOC_ORIENTATION=true \
OCR_USE_DOC_UNWARPING=true \
OCR_TEXT_DET_LIMIT_SIDE_LEN=4000 \
OCR_TEXT_DET_LIMIT_TYPE=max \
  python extract.py
mv output output_B_gpu_accurate && mkdir output
```

- Model/param names are **verified against the PaddleOCR 3.x constructor**:
  `PP-OCRv5_server_det`, `PP-OCRv5_server_rec`, and `SLANet_plus` bound to
  `wired_table_structure_recognition_model_name` (financial statements are almost
  always *wired*/bordered tables). There is no single `table_recognition_model_name`
  in 3.x — that was a wrong name and is now fixed in `extract.py`.
- The script prints `Using kwargs` / `Skipped` at startup. Because `PPStructureV3`
  accepts `**kwargs`, valid-but-unnamed options (`device`, `precision`, `cpu_threads`)
  pass through correctly rather than being wrongly dropped. If a model name is truly
  invalid the pipeline will raise at init — that's a loud failure, not a silent one.
- The first Run downloads model weights (one-time, cached in `~/.paddlex`).
- Note the printed `Total Time` and `Avg sec/page` for each run.

---

## 6. Verify the results (this is how we decide "good enough")

**Speed** — compare the printed `Avg sec/page`:
- CPU baseline (your existing run): fill in from your earlier output.
- Run A (GPU baseline) and Run B (GPU accurate): from the console.

**Accuracy** — there is no ground-truth file, so spot-check by hand:
```bash
# Pull both output sets back to the Mac for side-by-side review
gcloud compute scp --recurse $VM:~/output_A_gpu_baseline ./ --zone=$ZONE
gcloud compute scp --recurse $VM:~/output_B_gpu_accurate ./ --zone=$ZONE
```
1. Pick **3–5 dense-table pages** (financial statements — balance sheet, P&L notes).
2. Open the source page image next to `document.txt` / `tables.xlsx` for that page.
3. Field-check the **numbers** — a `6`↔`8`, `1`↔`7`, misplaced decimal, or a dropped
   thousands separator is a real defect in a financial doc. This is where server
   models should beat the CPU/mobile baseline.
4. Check **table structure** — merged cells, row/column alignment (SLANet_plus win).
5. Compare Run A vs Run B on the same pages to confirm the accuracy levers actually
   helped (and weren't silently skipped).

Optional quick text diff to surface where outputs differ:
```bash
diff <(cat output_A_gpu_baseline/document.txt) <(cat output_B_gpu_accurate/document.txt) | head -100
```

---

## 7. Teardown (⚠️ do this to stop billing)

```bash
# Option 1: stop (keeps disk, ~$0.10/day, fast to resume)
gcloud compute instances stop $VM --zone=$ZONE

# Option 2: delete entirely (no further cost)
gcloud compute instances delete $VM --zone=$ZONE
```
A Spot VM still bills while **running**. Stop or delete the moment testing is done.

---

## 8. Cost estimate (us-central1, approximate)

| Item | On-demand | Spot |
|---|---|---|
| n1-standard-4 | ~$0.19/hr | ~$0.04–0.08/hr |
| Tesla T4 ×1 | ~$0.35/hr | ~$0.11/hr |
| 50GB pd-ssd | ~$0.10/day | ~$0.10/day |
| **Effective / hr** | **~$0.55** | **~$0.20** |

A full test session (setup + a couple of runs) ≈ **1–2 hours → well under $1 on Spot.**

---

## 9. Risks & gotchas (in likelihood order)

1. **GPU quota = 0** on a fresh project → VM create fails. Handled in step 1d. Most likely blocker.
2. **Spot preemption** mid-run → VM stops; just `instances start` and re-run. Low risk for a 1–2 hr window.
3. **Model kwarg name mismatch** across paddleocr versions → the script's "Skipped"
   printout catches this; adjust env var to the correct param name and re-run.
4. **cu118 wheel doesn't see GPU** (rare) → fall back to cu126 wheel (step 3c note).
5. **First run is slow** due to one-time model download — not a real speed number; use
   the second run's timing.
6. **Zone has no T4 capacity** (Spot) → try `us-central1-b/-c/-f` or `us-west1-b`.

---

## 10. Decision after the test

- If Run B hits acceptable accuracy **and** speed → productionize: bake this VM/env
  into a reusable image or container, decide batch vs. on-demand serving.
- If accuracy still short on specific pages → consider targeted higher-DPI re-render
  of *those* pages only, or a different table model (`SLANeXt_wired`/`_wireless`).
- If GPU speed is fine but you want cheaper → test the same server models on a bigger
  CPU VM to see if accuracy alone (without GPU) is affordable for your volume.
