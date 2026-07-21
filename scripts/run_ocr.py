"""Full-book OCR pass over a PDF using Surya, cached one JSON file per page.

Resumable: skips any page whose cache file already exists, so an
interrupted run can just be re-invoked with the same range. Loads the
Surya model server once per process and keeps it warm for the whole
run instead of respawning per page.

Before running the full book, benchmark a handful of pages first --
throughput is decode-bound (tokens/sec per parallel slot), not a fixed
per-page cost, so a short smoke test on real content pages (not blank
front-matter pages) is the only reliable way to estimate total runtime
on a given machine. See README "Benchmark before committing" section.

Usage:
    python scripts/run_ocr.py book.pdf ocr_cache/ 1 794
    python scripts/run_ocr.py book.pdf ocr_cache/ 1 20   # smoke test first
"""
import argparse
import json
import time
from pathlib import Path

import fitz
from PIL import Image

from surya.inference import SuryaInferenceManager
from surya.recognition import RecognitionPredictor

DPI = 200
BATCH_SIZE = 8  # match RecognitionPredictor's default parallel slot count so
                 # cache files land incrementally instead of all-at-once


def page_to_image(page) -> Image.Image:
    pix = page.get_pixmap(dpi=DPI, colorspace=fitz.csRGB)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf_path")
    ap.add_argument("cache_dir")
    ap.add_argument("start_page", type=int, nargs="?", default=1)
    ap.add_argument("end_page", type=int, nargs="?", default=None)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(args.pdf_path)
    total = doc.page_count
    end_page = min(args.end_page or total, total)

    pending = [
        p for p in range(args.start_page, end_page + 1)
        if not (cache_dir / f"page_{p:04d}.json").exists()
    ]
    print(f"Total pages in range: {end_page - args.start_page + 1}, "
          f"already cached: {(end_page - args.start_page + 1) - len(pending)}, "
          f"remaining: {len(pending)}", flush=True)

    if not pending:
        print("Nothing to do.", flush=True)
        return

    manager = SuryaInferenceManager()
    rec_predictor = RecognitionPredictor(manager)

    t0 = time.time()
    done = 0
    for i in range(0, len(pending), BATCH_SIZE):
        batch_pages = pending[i:i + BATCH_SIZE]
        images = [page_to_image(doc[p - 1]) for p in batch_pages]

        results = rec_predictor(images, full_page=True)

        for p, result in zip(batch_pages, results):
            out = result.model_dump()
            out["source_page"] = p
            if not out.get("blocks"):
                # A dead/unresponsive llama-server (e.g. connection errors mid-run)
                # can make the driver return an empty result instead of raising.
                # Don't cache that as if the page succeeded -- leave it uncached
                # so a re-invocation retries it instead of silently skipping it.
                print(f"WARNING: page {p} returned no blocks -- not caching, "
                      f"will retry on next invocation", flush=True)
                continue
            with open(cache_dir / f"page_{p:04d}.json", "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False)

        done += len(batch_pages)
        elapsed = time.time() - t0
        rate = elapsed / done
        remaining = len(pending) - done
        eta_min = (remaining * rate) / 60
        print(f"[{done}/{len(pending)}] pages {batch_pages[0]}-{batch_pages[-1]} done. "
              f"{rate:.1f}s/page, ETA {eta_min:.1f} min", flush=True)


if __name__ == "__main__":
    main()
