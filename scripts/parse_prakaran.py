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

# Accept Gujarati AND Devanagari digits -- surya-ocr occasionally hallucinates
# Devanagari script (including digits) on pages with stylized/decorative
# typography even though the source is printed Gujarati. Without this, a
# misread marker like "१" (Devanagari) instead of "૧" (Gujarati) fails to
# parse as a number and silently corrupts downstream output -- see the
# verse_buffer anomaly handling below.
DIGIT_CHARS = "૦૧૨૩૪૫૬૭૮૯०१२३४५६७८९"
DIGIT_TRANSLATE = str.maketrans(DIGIT_CHARS, "0123456789" * 2)
NON_DIGIT_RE = re.compile(r"[^૦-૯०-९0-9]")

NOISE_EXACT = {
    "INDEX",
    "શ્રી મુક્તમુનિ ચરિત્ર ચિંતામણિ",
    "શ્રી મુક્તમુનિ યુરિત્ર ચિંતામણિ",  # OCR variant seen on some pages
}
NOISE_PATTERNS = [
    re.compile(r"^(?:પ્ર|प्र)\.\s*[૦-૯०-९]+$"),  # corner chapter marker, e.g. "પ્ર. ૨"
    re.compile(r"^[૦-૯०-९0-9]+$"),                # plain page number
]
# Also accept the Devanagari spelling "प्रकरण" for the same reason as the
# digit widening above -- the marker word itself can come back mis-scripted.
PRAKARAN_RE = re.compile(r"(?:પ્રકરણ|प्रकरण)\s*[:：]\s*([૦-૯०-९0-9]+)")
METER_LABEL_RE = re.compile(r"^([^\s:：]{1,15})\s*[:：]\s*$")

# When a page's verses render as a <p> with <br/>-separated lines instead of
# a <table> (an OCR layout inconsistency, not a content problem), the parser
# used to dump the whole thing as one undifferentiated "note" blob. Every
# doha/chopai couplet in this book is exactly 2 lines, and the book's fixed
# structure is: verses 1-3 are always પૂર્વછાયો, verses 4-25 are always
# ચોપાઈ, and whatever follows is prose (colophon) -- confirmed by the book's
# author/editor, not inferred. So instead of trusting the (often garbled)
# verse-end numeral OCR'd inline, split by position and assign meter by the
# fixed rule below.
BR_SPLIT_RE = re.compile(r"<br\s*/?>", re.I)
# Digit class here also accepts stray Odia digits (e.g. "୩") -- a rare
# cross-script OCR artifact seen even at Q8_0 quantization; harmless to
# strip since we ignore the marker's value entirely regardless.
TRAILING_MARKER_RE = re.compile(r"[॥|]+\s*[૦-૯०-९0-9୦-୯]*\s*[॥|]*\s*$")


def split_br_lines(html_str: str):
    """Split a <p>line<br/>line<br/>...</p> block into stripped text lines."""
    inner = re.sub(r"^<p[^>]*>|</p>\s*$", "", html_str.strip(), flags=re.I).strip()
    lines = []
    for seg in BR_SPLIT_RE.split(inner):
        t = (strip_tags(seg) if "<" in seg else seg).strip()
        if t:
            lines.append(t)
    return lines


def looks_like_verse_blob(lines) -> bool:
    """Verse lines end in a danda/pipe verse-end mark; colophon prose mostly
    doesn't (only its very last line does) -- use a majority threshold."""
    if len(lines) < 2:
        return False
    ending = sum(1 for l in lines if TRAILING_MARKER_RE.search(l))
    return ending / len(lines) >= 0.6


def meter_for_position(verse_num: int) -> str:
    return "પૂર્વછાયો" if verse_num <= 3 else "ચોપાઈ"


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


def to_int(num_str: str):
    try:
        return int(num_str.translate(DIGIT_TRANSLATE))
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
    verse_buffer = []            # accumulated lines for the verse in progress
    verse_buffer_start_page = None

    def target_list():
        return front_matter if current_prakaran is None else prakaran_items.setdefault(current_prakaran, [])

    def flush_verse_buffer_as_unparsed(reason):
        """Dump whatever's stuck in verse_buffer as a flagged entry instead of
        letting it silently carry across a prakaran boundary or get dropped.
        A verse only ends up here when its trailing marker never parsed as a
        number (e.g. a misread digit) -- content, not shape, is the problem."""
        nonlocal verse_buffer, verse_buffer_start_page
        if verse_buffer:
            target_list().append({
                "prakaran": current_prakaran,
                "type": "unparsed",
                "lines": verse_buffer,
                "source_page": verse_buffer_start_page,
                "note": reason,
            })
            print(f"WARNING: page {verse_buffer_start_page}: flushed "
                  f"{len(verse_buffer)} unparsed line(s) into prakaran "
                  f"{current_prakaran} ({reason})", flush=True)
            verse_buffer = []
            verse_buffer_start_page = None

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
                    if not verse_buffer:
                        verse_buffer_start_page = page_num
                    verse_buffer.append(line)
                    num = to_int(NON_DIGIT_RE.sub("", marker))
                    if num is not None:
                        target_list().append({
                            "prakaran": current_prakaran,
                            "verse_number": num,
                            "meter": current_meter,
                            "lines": verse_buffer,
                            "source_page": page_num,
                        })
                        verse_buffer = []
                        verse_buffer_start_page = None
                # leftover unflushed lines (no trailing number) carry over
                # to the next table/page as-is
                continue

            if "<br" in html_str.lower():
                br_lines = split_br_lines(html_str)
                if looks_like_verse_blob(br_lines):
                    clean = [TRAILING_MARKER_RE.sub("", l).strip() for l in br_lines]
                    clean = [l for l in clean if l]
                    for i in range(0, len(clean) - 1, 2):
                        vnum = i // 2 + 1
                        target_list().append({
                            "prakaran": current_prakaran,
                            "verse_number": vnum,
                            "meter": meter_for_position(vnum),
                            "lines": [clean[i], clean[i + 1]],
                            "source_page": page_num,
                        })
                    if len(clean) % 2 == 1:
                        print(f"WARNING: page {page_num}: odd leftover line "
                              f"in paragraph-verse blob: {clean[-1]!r}", flush=True)
                    continue

            text = strip_tags(html_str) if "<" in html_str else html_str.strip()
            if is_noise(text):
                continue

            prakaran_match = PRAKARAN_RE.search(text)
            if prakaran_match:
                new_prakaran = to_int(prakaran_match.group(1))
                flush_verse_buffer_as_unparsed(
                    f"verse marker never recognized before prakaran changed "
                    f"to {new_prakaran}")
                current_prakaran = new_prakaran
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

    flush_verse_buffer_as_unparsed("verse marker never recognized before end of OCR cache")
    if front_matter:
        (out_dir / "front-matter.json").write_text(
            json.dumps(front_matter, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    for num, items in sorted(prakaran_items.items()):
        out_path = out_dir / f"prakaran-{num}.json"
        out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(prakaran_items)} prakaran files, "
          f"{len(front_matter)} front-matter items.")


if __name__ == "__main__":
    main()
