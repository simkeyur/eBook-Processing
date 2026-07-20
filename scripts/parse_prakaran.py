"""Turn cached per-page OCR output into per-prakaran (chapter) JSON files.

Strips repeating page noise (running title, corner chapter marker, plain
page number, watermark labels), then walks blocks in page order tracking
the current prakaran (chapter) and meter (પૂર્વછાયો / ચોપાઈ / etc.) to
build one JSON object per verse.

This noise list and the meter-detection regex were tuned against one
specific book's layout -- re-validate against a few pages of a new book
before trusting the output (see probe_pdf.py + a manual eyeball pass).

Usage:
    python scripts/parse_prakaran.py ocr_cache/ prakaran_out/
"""
import argparse
import json
import re
from pathlib import Path
from html.parser import HTMLParser

GUJ_DIGITS = str.maketrans("૦૧૨૩૪૫૬૭૮૯", "0123456789")

NOISE_EXACT = {
    "INDEX",
    "શ્રી મુક્તમુનિ ચરિત્ર ચિંતામણિ",
    "શ્રી મુક્તમુનિ યુરિત્ર ચિંતામણિ",  # OCR variant seen on some pages
}
NOISE_PATTERNS = [
    re.compile(r"^પ્ર\.\s*[૦-૯]+$"),      # corner chapter marker, e.g. "પ્ર. ૨"
    re.compile(r"^[૦-૯0-9]+$"),           # plain page number
]
PRAKARAN_RE = re.compile(r"પ્રકરણ\s*[:：]\s*([૦-૯0-9]+)")
METER_LABEL_RE = re.compile(r"^([^\s:：]{1,15})\s*[:：]\s*$")


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def text(self):
        return "".join(self.parts).strip()


def strip_tags(html_str: str) -> str:
    ex = TextExtractor()
    ex.feed(html_str)
    return ex.text()


def to_int(guj_num: str):
    try:
        return int(guj_num.translate(GUJ_DIGITS))
    except ValueError:
        return None


def parse_table_rows(html_str: str):
    """Yield (line_text, marker_text) for each <tr> in a table block."""
    row_re = re.compile(r"<tr>(.*?)</tr>", re.S)
    cell_re = re.compile(r"<td>(.*?)</td>", re.S)
    for row_match in row_re.finditer(html_str):
        cells = cell_re.findall(row_match.group(1))
        if not cells:
            continue
        line = strip_tags(cells[0])
        marker = strip_tags(cells[1]) if len(cells) > 1 else ""
        yield line, marker


def is_noise(text: str) -> bool:
    if not text:
        return True
    if text in NOISE_EXACT:
        return True
    return any(p.match(text) for p in NOISE_PATTERNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir")
    ap.add_argument("out_dir")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = sorted(cache_dir.glob("page_*.json"))
    if not pages:
        print("No cached pages found.")
        return

    prakaran_items = {}     # prakaran_num -> list of verse/note dicts
    front_matter = []       # content before the first પ્રકરણ marker

    current_prakaran = None
    current_meter = None
    verse_buffer = []       # accumulated lines for the verse in progress

    def target_list():
        return front_matter if current_prakaran is None else prakaran_items.setdefault(current_prakaran, [])

    for page_path in pages:
        data = json.loads(page_path.read_text(encoding="utf-8"))
        page_num = data.get("source_page")
        blocks = data.get("blocks", [])

        for block in blocks:
            html_str = block.get("html") or block.get("text") or ""
            is_table = "<table" in html_str

            if is_table:
                for line, marker in parse_table_rows(html_str):
                    if not line:
                        continue
                    verse_buffer.append(line)
                    num = to_int(re.sub(r"[^૦-૯0-9]", "", marker))
                    if num is not None:
                        target_list().append({
                            "prakaran": current_prakaran,
                            "verse_number": num,
                            "meter": current_meter,
                            "lines": verse_buffer,
                            "source_page": page_num,
                        })
                        verse_buffer = []
                # leftover unflushed lines (no trailing number) carry over
                # to the next table/page as-is
                continue

            text = strip_tags(html_str) if "<" in html_str else html_str.strip()
            if is_noise(text):
                continue

            prakaran_match = PRAKARAN_RE.search(text)
            if prakaran_match:
                current_prakaran = to_int(prakaran_match.group(1))
                current_meter = None
                continue

            meter_match = METER_LABEL_RE.match(text)
            if meter_match:
                current_meter = meter_match.group(1)
                continue

            # other prose: chapter intro/colophon lines, captions, etc.
            target_list().append({
                "prakaran": current_prakaran,
                "type": "note",
                "text": text,
                "source_page": page_num,
            })

    for num, items in sorted(prakaran_items.items()):
        out_path = out_dir / f"prakaran-{num}.json"
        out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    if front_matter:
        (out_dir / "front-matter.json").write_text(
            json.dumps(front_matter, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"Wrote {len(prakaran_items)} prakaran files, "
          f"{len(front_matter)} front-matter items.")
    if verse_buffer:
        print(f"WARNING: {len(verse_buffer)} unflushed verse line(s) at end of run: {verse_buffer}")


if __name__ == "__main__":
    main()
