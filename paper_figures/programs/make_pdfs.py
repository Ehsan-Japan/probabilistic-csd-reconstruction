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
            print("  %-46s %s pages   %d/%d pictures at full resolution"
                  % (name, info, n_ok, n_all))
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
