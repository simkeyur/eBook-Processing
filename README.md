# eBook Processing — Gujarati/Sanskrit PDF → structured JSON

Pipeline for turning a scanned/printed Gujarati + Sanskrit religious text PDF
(chopai/doha verse format, organized into પ્રકરણ / chapters) into clean,
per-chapter JSON suitable for serving from a web app.

Source PDF is **not** included in this repo (large, copyrighted scanned
book). Drop your own PDF in and point the scripts at it.

## Pipeline

```
probe_pdf.py  →  run_ocr.py  →  parse_prakaran.py
 (sanity check)   (OCR, cached      (strip noise, group
                   per page)         into per-chapter JSON)
```

1. **`probe_pdf.py`** — before committing to a long OCR run, check the PDF's
   page count and render a few sample pages so you can eyeball print quality.
2. **`run_ocr.py`** — runs [Surya OCR](https://github.com/VikParuchuri/surya)
   over a page range, caching one JSON file per page (resumable).
3. **`parse_prakaran.py`** — reads the cached per-page OCR output, strips
   repeating page noise (running title, corner page markers, watermark),
   and groups verses into one JSON file per chapter (`prakaran-N.json`).

See `examples/` for real sample output from two chapters of the source book
(pages 48 and 397) at every stage of the pipeline, so you can see the
expected shape of the data without running anything.

## Key findings

- **The PDF's embedded text layer is not usable.** `pymupdf`/`PyMuPDF`
  reports non-zero extractable text on most pages, but it decodes to
  garbage (e.g. `©e {wõ‚{wr™ [rhºt`) — the PDF was produced with a legacy
  non-Unicode Gujarati font, so the underlying bytes map to font glyph
  indices, not real Unicode codepoints. **Always OCR from the rendered
  page image, never trust `page.get_text()` on older Indic-script PDFs**
  just because it returns a non-empty string — verify against the
  rendered image first (`probe_pdf.py` does this for you).

- **Surya OCR handles Gujarati + Devanagari (Sanskrit) well.** Verified
  against two full chapters — every verse, matra, and conjunct cluster
  matched the source image exactly, with only the occasional misread on
  tiny decorative corner numerals (not the actual verse text). Other
  models considered and ruled out: GOT-OCR2.0 and docTR are tuned mainly
  for Latin/CJK and are unreliable on Indic scripts; Tesseract's `guj`/`san`
  trained data is a viable fallback if Surya is ever unavailable.

- **OCR throughput is decode-bound, not compute-bound, and per-page cost
  varies enormously by content density.** Surya's `full_page=True` mode
  asks the model to emit the entire page as HTML-tagged text (tables,
  paragraph tags, etc.), which is verbose — a dense poetry page runs
  ~2000 output tokens. At ~12-13 tokens/sec per parallel slot (8 slots
  default) that's ~150-200s per *content-heavy* page, vs. near-instant for
  a blank front-matter/title page. **Don't estimate total runtime from a
  smoke test on the first few pages of a book** — front matter pages are
  usually far lighter than body content and will give a wildly optimistic
  ETA. Benchmark on real body-text pages instead (see below).

- **Hardware notes.** This workload is memory-bandwidth-bound (typical for
  autoregressive decode), not raw-FLOPS-bound. An Apple M4's unified
  memory bandwidth (~120GB/s) comfortably beats a Jetson Orin Nano 8GB's
  (~68GB/s) — moving this pipeline to a Jetson is about freeing up your
  primary machine / running unattended for hours, not necessarily a speed
  win. Don't assume newer/GPU-labeled hardware is automatically faster for
  this kind of job; benchmark before committing.

## Gotchas (things that cost us real time)

- **`HF_HUB_OFFLINE=1`** — once model weights are downloaded and cached
  (`~/.cache/huggingface/hub`), set this env var. Without it, every run
  does a network metadata check against the HF Hub before starting, which
  is slow and unnecessary once the weights are local.

- **Surya reuses a running `llama-server` across process launches** via a
  sentinel file at `~/.cache/datalab/surya/llamacpp_server.json` (probe
  `/health` → reuse if alive, else spawn fresh). This is normally a nice
  optimization (no cold-start cost on every invocation) but has two traps:
  - Killing the Python driver process does **not** kill the spawned
    `llama-server` child — you'll leak an orphaned GPU/CPU-resident
    process. Always kill both, or `pkill llama-server` explicitly.
  - If you kill the shared `llama-server` while a *different* driver
    process is still relying on it (via the sentinel), that driver starts
    failing with `Inference error: Connection error.` on every page. Clear
    the stale sentinel (`rm ~/.cache/datalab/surya/llamacpp_server.json*`)
    before restarting.

- **Batch size controls write granularity, not just throughput.** The OCR
  driver only writes cache files after an entire batch finishes. A batch
  size larger than the model server's parallel slot count (default 8)
  means long stretches with zero visible progress on disk even though the
  GPU is actively working — `run_ocr.py` defaults `BATCH_SIZE = 8` to
  match the parallel slot count so pages land on disk as each one finishes.

- **Sustained parallel decode workloads generate real heat** on a laptop.
  Running 8 parallel OCR slots for hours is a legitimate reason to move
  this to dedicated/fan-cooled hardware rather than a MacBook.

## Setup

### macOS

```bash
brew install llama.cpp        # provides the llama-server binary (Metal backend)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Linux / Jetson Orin Nano (agent runbook)

This section is written as a step-by-step runbook for an AI coding agent
(or a human) setting this up on a fresh Linux box. No Homebrew on
aarch64/Jetson, so `llama-server` needs to be built from source with CUDA
support.

```bash
# 1. System deps
sudo apt update && sudo apt install -y build-essential cmake git python3-venv python3-pip

# 2. Build llama.cpp with CUDA support (Jetson ships CUDA via JetPack already --
#    verify with `nvcc --version` first; install JetPack if missing)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j"$(nproc)"
# Put the built binary on PATH, or set LLAMA_CPP_BINARY to its path:
export LLAMA_CPP_BINARY="$(pwd)/build/bin/llama-server"
cd ..

# 3. Python env for this project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# NOTE: surya-ocr pulls in torch (+torchvision) as a transitive dependency, but
# this pipeline's actual OCR inference runs through the separate llama-server
# process (CUDA via llama.cpp), not through torch tensor ops -- so a CPU-only
# torch wheel is all that's needed. Be aware the DEFAULT PyPI linux torch wheel
# bundles ~1GB of NVIDIA CUDA libraries; requirements.txt pins the +cpu wheels
# from download.pytorch.org/whl/cpu to avoid that (see "Linux / x86_64 CPU"
# below). On a real Jetson you also do NOT need NVIDIA's special JetPack
# PyTorch wheel -- the CPU wheel is fine since torch isn't in the hot path.

# 4. Sanity-check the PDF before doing anything expensive
python scripts/probe_pdf.py your_book.pdf --pages 1,50,100 --dpi 200 --out-dir probe_out
# -> eyeball probe_out/*.png against the printed "Embedded text sample" --
#    if they don't match, that confirms you must OCR from images (expected
#    for this book; see Key findings above).

# 5. Benchmark before committing to the full book -- pick body-text pages,
#    not front matter, and time it for real:
time HF_HUB_OFFLINE=1 python scripts/run_ocr.py your_book.pdf ocr_cache/ 48 55
# extrapolate (total_pages / benchmarked_pages) * elapsed_seconds for an
# honest ETA before kicking off the full run.

# 6. Full run (resumable -- safe to Ctrl-C and re-invoke with the same range)
HF_HUB_OFFLINE=1 python scripts/run_ocr.py your_book.pdf ocr_cache/ 1 794

# 7. Parse into per-chapter JSON once OCR is done (or partially done --
#    it'll just process whatever pages are cached so far)
python scripts/parse_prakaran.py ocr_cache/ prakaran_out/
```

**Before declaring the setup done, the agent should:** run step 4 and 5,
report the extrapolated full-book ETA back to the user, and get
confirmation before starting step 6 unattended for hours — see the
"Gotchas" section above on why a fast smoke test can be misleading.

### Linux / x86_64 CPU (AMD or Intel, no NVIDIA GPU)

For an ordinary x86_64 Linux box with no NVIDIA GPU — including AMD APU
machines (e.g. Ryzen + integrated Radeon Vega) — build `llama-server`
**CPU-only**. Do **not** use `GGML_CUDA=ON` (no CUDA) and don't bother with
the AMD iGPU via ROCm/HIP either: an integrated GPU shares the CPU's memory
bus, and this workload is memory-bandwidth-bound (see "Hardware notes"
above), so the iGPU offers no speedup over the CPU while ROCm on unsupported
`gfx90c`-class APUs is fragile. The multi-core CPU build is the reliable,
equal-or-faster choice here.

```bash
# 1. System deps. If you don't have root, cmake is also available as a pip
#    wheel (`pip install cmake`) -- everything else (gcc/g++/make/git) is
#    usually already present on a dev box.
sudo apt update && sudo apt install -y build-essential cmake git python3-venv python3-pip

# 2. Build llama.cpp CPU-only (note GGML_CUDA=OFF, and CURL off to avoid an
#    extra system dep). cmake auto-detects -march=native + OpenMP.
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
cmake --build build --config Release -j"$(nproc)" --target llama-server
export LLAMA_CPP_BINARY="$(pwd)/build/bin/llama-server"
cd ..

# 3. Python env (same as the Jetson runbook). requirements.txt already pins
#    the CPU-only torch/torchvision wheels on Linux, so this pulls no CUDA
#    libraries.
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 4-7. Probe / benchmark / full run / parse -- identical to the Jetson runbook
#      steps above. Benchmark on real body-text pages, not front matter.
```

## Usage reference

```bash
# Probe a PDF
python scripts/probe_pdf.py <pdf_path> [--pages 1,48,397] [--dpi 200] [--out-dir probe_out]

# OCR a page range (resumable)
python scripts/run_ocr.py <pdf_path> <cache_dir> [start_page] [end_page]

# Parse cached OCR output into per-chapter JSON
python scripts/parse_prakaran.py <cache_dir> <out_dir>
```

## Output schema

One JSON object per verse in each `prakaran-N.json`:

```json
{
  "prakaran": 2,
  "verse_number": 4,
  "meter": "ચોપાઈ",
  "lines": [
    "ધન્ય ધન્ય તે કૌશલ દેશ, એમાં અવધપુર વિશેષ",
    "ધન્ય ધન્ય તે અજય તાત, ધન્ય ધન્ય તે સુમતિ માત"
  ],
  "source_page": 48
}
```

Content encountered before the first `પ્રકરણ : N` marker (title page,
dedication, photo captions) is written to `front-matter.json` instead of
being forced into a chapter.

## Known limitations

- Noise-stripping patterns (`parse_prakaran.py`) were tuned against this
  specific book's layout (title string, corner marker format). Re-validate
  against a few pages before reusing on a different book.
- Meter-label detection is a generic regex (`label :` on its own line) —
  works for `પૂર્વછાયો`/`ચોપાઈ`, untested at scale against books that mix in
  `દોહા`, `સોરઠા`, or Sanskrit `શ્લોક` sections.
- OCR occasionally misreads small decorative corner numerals (e.g. chapter
  number in the page header graphic); the actual verse text and the
  `પ્રકરણ : N` heading elsewhere on the page have been reliable in testing.
