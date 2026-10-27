# -*- coding: utf-8 -*-
"""
make_pdfs.py -- every .docx under method_docs/ exported to .pdf by Word.

The PDF is never built from the generator: it is exported from whatever
.docx is on disk.  So a document you edited by hand in Word is exported
exactly as you left it, and the PDF beside it is always the same document
as the Word file, not an older machine-made version of it.

    python paper_figures/programs/make_pdfs.py          # only what changed
    python paper_figures/programs/make_pdfs.py --all    # everything

A document open in Word is skipped and named, rather than failing the run.
Windows only: it drives the installed Word through PowerShell.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
OUT_DIR = os.path.join(PKG, "method_docs")

# wdExportFormatPDF
PS = r"""
$ErrorActionPreference = 'Stop'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$failed = @()
foreach ($src in $args) {
  $pdf = [System.IO.Path]::ChangeExtension($src, '.pdf')
  try {
    $doc = $word.Documents.Open($src, $false, $true)
    $n = $doc.ComputeStatistics(2)
    $doc.ExportAsFixedFormat($pdf, 17)
    $doc.Close(0)
    Write-Output ("OK`t{0}`t{1}" -f [System.IO.Path]::GetFileName($src), $n)
  } catch {
    Write-Output ("SKIP`t{0}`t{1}" -f [System.IO.Path]::GetFileName($src), $_.Exception.Message)
  }
}
$word.Quit()
"""


def _docx_pictures(docx_path):
    """The pictures of a .docx, in the order the body shows them."""
    import re
    import zipfile
    z = zipfile.ZipFile(docx_path)
    rels = z.read("word/_rels/document.xml.rels").decode("utf-8")
    target = dict(re.findall(r'Id="([^"]+)"[^>]*Target="([^"]+)"', rels))
    target.update({k: v for v, k in
                   re.findall(r'Target="([^"]+)"[^>]*Id="([^"]+)"', rels)})
    body = z.read("word/document.xml").decode("utf-8")
    out = []
    for rid in re.findall(r'r:embed="([^"]+)"', body):
        name = "word/" + target[rid].lstrip("/").replace("word/", "", 1)
        out.append(z.read(name))
    return out


def restore_full_resolution(docx_path):
    """Put the .docx's own pictures back into the PDF Word exported.

    Word's PDF writer downsamples every picture to 200 dpi, whatever the
    document's compression settings say (tested 2026-09-30: the 3420-px
    Fig. 13 came out at 1219 px with doNotAutoCompressPictures, and with
    defaultImageDpi 0 or 600).  The .docx holds the full image, so each
    picture of the PDF is replaced by the one it was made from: the two are
    walked in order and paired only when their aspect ratios agree.
    Returns (replaced, pictures in the PDF)."""
    import io
    import pymupdf
    from PIL import Image
    pdf_path = os.path.splitext(docx_path)[0] + ".pdf"
    blobs = _docx_pictures(docx_path)
    aspects = [(lambda im: im.size[0] / im.size[1])(Image.open(io.BytesIO(b)))
               for b in blobs]
    doc = pymupdf.open(pdf_path)
    seen, k, done, total = set(), 0, 0, 0
    for page in doc:
        infos = sorted((i for i in page.get_image_info(xrefs=True)
                        if i["xref"] and i["xref"] not in seen),
                       key=lambda i: (i["bbox"][1], i["bbox"][0]))
        for info in infos:
            seen.add(info["xref"])
            total += 1
            a = info["width"] / info["height"]
            j = k
            while j < len(blobs) and abs(aspects[j] / a - 1) > 0.02:
                j += 1
            if j == len(blobs):
                continue
            page.replace_image(info["xref"], stream=blobs[j])
            k, done = j + 1, done + 1
    tmp = pdf_path + ".tmp"
    doc.save(tmp, garbage=3, deflate=True)
    doc.close()
    os.replace(tmp, pdf_path)
    return done, total


CAPTION_RE = r"^(Fig\.|Table)\s*([0-9IVX]+)\."
MENTION_RE = r"\bFig(?:ure)?s?\.?\s+(\d+)"
BOLD_REF_RE = r"Fig(?:ure)?s?\.?\s*(\d+)|Tables?\s*([IVX]+)\b"


def add_figure_links(pdf_path):
    """Make every figure and table in the PDF a link to the first place the
    main text mentions it, and every bold "Figure N" / "Table N" of the
    text a link to that figure (its picture) or table (its caption).

    A figure is the picture just above its caption ("Fig. N." starting a
    text block); the main text is everything before the page of the first
    caption, so captions that cite another figure do not count.  The link
    lands a little above the mention.  For a table only its caption label
    ("Table III.") is linked, so that its text stays selectable.  Returns (figures and
    tables linked, figures and tables, bold references linked)."""
    import re
    import pymupdf
    doc = pymupdf.open(pdf_path)
    captions = {}                                  # N -> (page, caption rect)
    tables = {}                                    # "III" -> (page, rect)
    for page in doc:
        for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
            m = re.match(CAPTION_RE, text.strip())
            if m and m.group(1) == "Fig." and m.group(2).isdigit():
                captions.setdefault(int(m.group(2)),
                                    (page.number, pymupdf.Rect(x0, y0, x1, y1)))
            elif m and m.group(1) == "Table":
                tables.setdefault(m.group(2),
                                  (page.number, pymupdf.Rect(x0, y0, x1, y1)))
    if not captions:
        doc.close()
        return 0, 0, 0
    first_fig_page = min(pn for pn, _ in captions.values())
    mentions = {}                                  # N -> (page, point)
    for page in doc:
        if page.number >= first_fig_page:
            break
        for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
            if re.match(CAPTION_RE, text.strip()):
                continue
            for m in re.finditer(MENTION_RE, text):
                n = int(m.group(1))
                if n in mentions or n not in captions:
                    continue
                block = pymupdf.Rect(x0, y0, x1, y1)
                hits = [r for r in page.search_for(m.group(0).split()[0])
                        if block.intersects(r)] or [block]
                mentions[n] = (page.number, hits[0].tl)
    linked = 0
    pictures = {}                                  # N -> (page, picture rect)
    for n, (pn, cap) in captions.items():
        page = doc[pn]
        # the caption's text block starts on the blank line above it,
        # which the picture can overlap: compare the picture's top
        above = [pymupdf.Rect(i["bbox"]) for i in page.get_image_info()
                 if i["bbox"][1] < cap.y0]
        if not above:
            continue
        img = max(above, key=lambda r: r.y0)       # the picture over it
        pictures[n] = (pn, img)
        if n not in mentions:
            continue
        tp, pt = mentions[n]
        page.insert_link({"kind": pymupdf.LINK_GOTO, "from": img,
                          "page": tp, "to": pymupdf.Point(0, max(pt.y - 30, 0)),
                          "zoom": 0})
        linked += 1

    # bold references in the text (not in a caption) -> the figure or table
    refs = 0
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            spans = [sp for ln in block.get("lines", ())
                     for sp in ln["spans"]]
            text = "".join(sp["text"] for sp in spans).strip()
            if not spans or re.match(CAPTION_RE, text):
                continue
            for sp in spans:
                if not (sp["flags"] & 16 or "Bold" in sp["font"]):
                    continue
                for m in re.finditer(BOLD_REF_RE, sp["text"]):
                    if m.group(1) and int(m.group(1)) in pictures:
                        tp, r = pictures[int(m.group(1))]
                    elif m.group(2) and m.group(2) in tables:
                        tp, r = tables[m.group(2)]
                    else:
                        continue
                    hits = page.search_for(m.group(0),
                                           clip=pymupdf.Rect(sp["bbox"]))
                    for h in hits or [pymupdf.Rect(sp["bbox"])]:
                        page.insert_link({"kind": pymupdf.LINK_GOTO,
                                          "from": h, "page": tp,
                                          "to": pymupdf.Point(
                                              0, max(r.y0 - 20, 0)),
                                          "zoom": 0})
                    refs += 1
    # each table (caption and rows) -> its first mention in the text
    for name, (pn, cap) in tables.items():
        first = None
        for page in doc:
            if page.number >= pn:
                break
            for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
                if re.match(CAPTION_RE, text.strip()):
                    continue
                m = re.search(r"\bTables?\s+" + name + r"\b", text)
                if m:
                    block = pymupdf.Rect(x0, y0, x1, y1)
                    hits = [r for r in page.search_for("Table")
                            if block.intersects(r)] or [block]
                    first = (page.number, hits[0].tl)
                    break
            if first:
                break
        if not first:
            continue
        page = doc[pn]
        # only the caption label ("Table III.") is the link: a link over the
        # caption or the rows would stop their text from being selected,
        # e.g. to comment on it
        label = [r for r in page.search_for("Table", clip=cap)
                 if abs(r.y0 - cap.y0) < 30]
        if not label:
            continue
        area = label[0]
        num = page.search_for(name + ".", clip=pymupdf.Rect(
            area.x1, area.y0 - 2, area.x1 + 40, area.y1 + 2))
        if num:
            area = area | num[0]
        tp, pt = first
        page.insert_link({"kind": pymupdf.LINK_GOTO, "from": area,
                          "page": tp, "to": pymupdf.Point(0, max(pt.y - 30, 0)),
                          "zoom": 0})
        linked += 1
    doc.saveIncr()
    doc.close()
    return linked, len(captions) + len(tables), refs


HEADING_RE = r"^(\d+(?:\.\d+)*)\.\s+[A-Z]"
SEC_REF_RE = r"Secs?\.[\s\u202f]*(\d+(?:\.\d+)*)"


def add_section_links(pdf_path):
    """Make every "Sec. 4.3" of the text a link to the heading of Sec. 4.3.

    A heading is a short text block starting "4.3. " and a capital letter;
    the link lands a little above it.  Returns (linked, headings)."""
    import re
    import pymupdf
    doc = pymupdf.open(pdf_path)
    heads = {}                                     # "4.3" -> (page, rect)
    for page in doc:
        for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
            t = text.strip()
            m = re.match(HEADING_RE, t)
            if m and len(t) < 120:
                heads.setdefault(m.group(1),
                                 (page.number, pymupdf.Rect(x0, y0, x1, y1)))
    linked = 0
    for page in doc:
        for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
            t = text.strip()
            if re.match(HEADING_RE, t) and len(t) < 120:
                continue
            block = pymupdf.Rect(x0, y0, x1, y1)
            for m in re.finditer(SEC_REF_RE, text):
                if m.group(1) not in heads:
                    continue
                tp, r = heads[m.group(1)]
                # the reference on the page: "Sec." and the number after it
                secs = page.search_for("Sec", clip=block)
                nums = page.search_for(m.group(1), clip=block)
                hit = None
                for sr in secs:
                    after = [nr for nr in nums
                             if abs(nr.y0 - sr.y0) < 3 and 0 <= nr.x0 - sr.x1 < 25]
                    if after and not any(
                            lk["from"].contains(sr.tl + (1, 1))
                            for lk in page.get_links()):
                        hit = sr | after[0]
                        break
                if hit is None:
                    continue
                page.insert_link({"kind": pymupdf.LINK_GOTO, "from": hit,
                                  "page": tp,
                                  "to": pymupdf.Point(0, max(r.y0 - 20, 0)),
                                  "zoom": 0})
                linked += 1
    doc.saveIncr()
    doc.close()
    return linked, len(heads)


def stale(docx_path):
    """True if the PDF is missing or older than the Word file."""
    pdf = os.path.splitext(docx_path)[0] + ".pdf"
    if not os.path.exists(pdf):
        return True
    return os.path.getmtime(docx_path) > os.path.getmtime(pdf)


def documents(everything=False):
    found = []
    # The assembled manuscript sits at the top of method_docs/ rather than
    # in a part folder.  It is named, not globbed by extension: the other
    # loose .docx files up there are the journal's templates and old
    # fragments, and they do not want a PDF beside them.
    for name in sorted(os.listdir(OUT_DIR)):
        if not name.startswith("Manuscript") or not name.endswith(".docx"):
            continue
        if name.startswith("~$") or "template" in name.lower():
            continue
        path = os.path.join(OUT_DIR, name)
        if everything or stale(path):
            found.append(path)
    for part in sorted(os.listdir(OUT_DIR)):
        d = os.path.join(OUT_DIR, part)
        # backups/ is history -- the manuscript's sits at the top of
        # method_docs/ and would otherwise look like a part folder.
        if (not os.path.isdir(d) or part.startswith(".")
                or part in ("backups", "_trash")):
            continue
        for name in sorted(os.listdir(d)):
            # backups/ is history; it does not need a PDF
            if not name.endswith(".docx") or name.startswith("~$"):
                continue
            path = os.path.join(d, name)
            if everything or stale(path):
                found.append(path)
    return found


def main():
    everything = "--all" in sys.argv
    todo = documents(everything)
    if not todo:
        print("every PDF is already newer than its .docx  (--all to force)")
        return []
    script = os.path.join(HERE, "_export_pdf.ps1")
    with open(script, "w", encoding="utf-8") as fh:
        fh.write(PS)
    out = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", script] + todo,
        capture_output=True, text=True)
    os.remove(script)
    done, skipped = [], []
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        tag, name, info = (line.split("\t") + ["", ""])[:3]
        if tag == "OK":
            src = next(p for p in todo if os.path.basename(p) == name)
            n_ok, n_all = restore_full_resolution(src)
            n_ln, n_fig, n_ref = add_figure_links(
                os.path.splitext(src)[0] + ".pdf")
            n_sec, _ = add_section_links(os.path.splitext(src)[0] + ".pdf")
            print("  %-46s %s pages   %d/%d pictures at full resolution"
                  "   %d/%d figures and tables linked to their first mention"
                  "   %d bold references linked   %d Sec. references linked"
                  % (name, info, n_ok, n_all, n_ln, n_fig, n_ref, n_sec))
            done.append(name)
        else:
            skipped.append((name, info))
    if out.returncode and not done:
        print(out.stderr.strip()[:800])
    if skipped:
        print()
        print("NOT exported (open in Word?  close it and re-run):")
        for name, why in skipped:
            print("   ", name, "--", why.split("\n")[0][:90])
    print()
    print("%d PDFs -> %s" % (len(done), OUT_DIR))
    return done


if __name__ == "__main__":
    main()
