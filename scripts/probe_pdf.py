"""Sanity-check a PDF before committing to a full OCR run.

Reports page count and, for each requested page: whether the PDF already
has a usable embedded text layer (many older Gujarati/Indic PDFs embed
text in a non-Unicode font, which decodes to garbage and is *not* usable
even though pymupdf reports non-zero text length -- always eyeball the
rendered image against the extracted text before trusting it), and
renders the page to a PNG so you can visually confirm print quality.

Usage:
    python scripts/probe_pdf.py book.pdf --pages 1,48,397 --dpi 200 --out-dir examples
"""
import argparse
from pathlib import Path

import fitz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf_path")
    ap.add_argument("--pages", default="1", help="Comma-separated 1-indexed page numbers")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out-dir", default="probe_out")
    args = ap.parse_args()

    doc = fitz.open(args.pdf_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pages: {doc.page_count}")
    print(f"Metadata: {doc.metadata}")

    for p in [int(x) for x in args.pages.split(",")]:
        page = doc[p - 1]
        text = page.get_text().strip()
        pix = page.get_pixmap(dpi=args.dpi, colorspace=fitz.csRGB)
        out_path = out_dir / f"page_{p:04d}.png"
        pix.save(out_path)

        print(f"\n--- Page {p} ---")
        print(f"Embedded text length: {len(text)}")
        print(f"Embedded text sample: {text[:150]!r}")
        print(f"Rendered: {out_path} ({pix.width}x{pix.height})")
        print("NOTE: non-zero embedded text length does NOT mean it's usable -- "
              "open the PNG and compare by eye. Legacy non-Unicode fonts decode "
              "to garbage bytes that look like real text length but aren't.")


if __name__ == "__main__":
    main()
