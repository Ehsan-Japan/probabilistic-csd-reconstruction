"""
make_table3_explained.py -- a short PDF explaining what Table III of
Manuscript_ver3 means.

    python paper_figures/programs/make_table3_explained.py

Every number is read from results/<run>/two_step/table.csv, the same file
the manuscript table is built from.  Writes Table_III_explained.pdf into
paper_figures/method_docs_ver2/.
"""
import csv
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUN = "4-5-6-7-8_rays_40-50-60_points_500_samples"
EVID = os.path.join(ROOT, "results", RUN, "two_step")
OUT = os.path.join(ROOT, "paper_figures", "method_docs_ver2")
BUILD = os.path.join(OUT, "table3_explained_build")


def main():
    with open(os.path.join(EVID, "table.csv"), newline="",
              encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def g(method, setting=""):
        return next(r for r in rows if r["method"] == method
                    and r["setting"].startswith(setting))
    def raster(matched):
        return next(r for r in rows if r["method"] == "coarse scan"
                    and r["matched"] == matched)
    R = {"U": g("U-Net only"), "CU": raster("U-Net only"),
         "B3": g("U-Net, then local check", "band ±3"),
         "B4": g("U-Net, then local check", "band ±4"),
         "B5": g("U-Net, then local check", "band ±5"),
         "B6": g("U-Net, then local check", "band ±6"),
         "CB5": raster("band ±5 px"), "F": g("full scan")}

    def m(k):
        return f"{float(R[k]['measured']):.1f}"

    def v(k, c):
        return f"{float(R[k][c]):.3f}"

    def side(k):
        n = R[k]["setting"].split(" x ")[0]
        return f"{n} $\times$ {n}"
    labels = [("U", "U-Net only"), ("CU", "Raster " + side("CU")),
              ("B3", "U-Net + local check, $\pm$3 px"),
              ("B4", "U-Net + local check, $\pm$4 px"),
              ("B5", "U-Net + local check, $\pm$5 px"),
              ("B6", "U-Net + local check, $\pm$6 px"),
              ("CB5", "Raster " + side("CB5")), ("F", "Full scan")]
    body = []
    for n, (k, lab) in enumerate(labels, 1):
        body.append(f"{n} & {lab} & {m(k)} & {v(k, 'precision')} & "
                    f"{v(k, 'recall')} & {v(k, 'f1')} \\\\")
        if n in (2, 7):
            body.append(r"\midrule")
    d = float(R["B5"]["f1"]) - float(R["F"]["f1"])
    vs = ("higher than" if d >= 0.005 else "lower than" if d <= -0.005
          else "about equal to")
    tex = r"""\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{lmodern}
\usepackage[margin=2.3cm]{geometry}\usepackage{booktabs}\usepackage{parskip}
\usepackage{enumitem}\setlist{itemsep=2pt}
\begin{document}
\begin{center}{\Large What Table III means}\[3pt]
\small Manuscript\_ver3, Sec.~4.2--4.3; 8 rays $\times$ 60 points, 50 test
devices\end{center}

\section*{The table}
\begin{center}
\begin{tabular}{rlrrrr}\toprule
 & method & measured (\%) & precision & recall & F1@1 \\midrule
""" + "\n".join(body) + r"""
\bottomrule\end{tabular}
\end{center}

\section*{What each row is}
Every row is a way of finding the transition lines of the same 50 test
devices. They differ in \emph{what is measured} and \emph{how the lines are
found}.
\begin{enumerate}
\item \textbf{U-Net only}: the device is measured along 8 rays of 60 points
  (""" + m("U") + r"""\,\% of the plane), and the U-Net predicts the lines.
\item \textbf{Raster """ + side("CU") + r"""}: the conventional measurement,
  a regular grid with at least as many points as row 1, then line
  detection.
\item[3--6.] \textbf{U-Net + local check}: the two-step method. After the
  U-Net has predicted the lines, the device is measured again in a strip
  around each predicted line, extending 3 to 6 pixels on either side. A
  predicted line is kept only where the newly measured signal really shows
  a step; only measured pixels enter this decision. The strip is the extra
  cost.
\item[7.] \textbf{Raster """ + side("CB5") + r"""}: a regular grid with at
  least as many points as row 5, for a fair comparison at that cost.
\item[8.] \textbf{Full scan}: every pixel measured, then the same line
  detection. This is the reference.
\end{enumerate}
The raster size is not free: $k$ is the smallest integer with $k^2$ at
least the number of pixels the U-Net setting measures, so the raster never
has fewer measurements. The line detection in rows 2, 7 and 8 is the same
step detector as in rows 3--6; its threshold was chosen on the 75
validation devices, never on the test devices.

\section*{What each column means}
\begin{itemize}
\item \textbf{Measured (\%)}: the fraction of the $100\times100$ grid that
  was measured. Lower means faster.
\item \textbf{Precision}: of all the pixels a method marked as a line, the
  fraction that really lie on a true line (within one pixel). Low precision
  means many \emph{false} lines.
\item \textbf{Recall}: of all the true line pixels, the fraction that the
  method found (within one pixel). Low recall means \emph{missed} lines.
\item \textbf{F1@1}: a single score combining both,
  $2PR/(P+R)$; 1 is perfect. The ``@1'' means a line found one pixel away
  from the truth still counts as found.
\end{itemize}

\section*{How to read it}
\textbf{Rows 1--2: the same number of measurements.} With """ + m("U") + r"""\,\%
of the plane measured, the U-Net reaches F1@1 = """ + v("U", "f1") + r""";
the raster reaches """ + v("CU", "f1") + r""". The raster finds many of the
true lines (recall """ + v("CU", "recall") + r"""), but only
""" + v("CU", "precision") + r""" of what it marks is a real line: between
its widely spaced points nothing is measured. The U-Net has learned what a
honeycomb looks like and places the lines between the rays.

\textbf{Rows 3--6: what the U-Net output is useful for.} The predicted lines
tell where to measure next. Measuring a strip around them and keeping only
the lines that the signal confirms raises F1@1 from """ + v("U", "f1") + r"""
to """ + v("B5", "f1") + r""" ($\pm$5 px, """ + m("B5") + r"""\,\% measured).
The gain is mostly in precision (""" + v("U", "precision") + r""" $\to$
""" + v("B5", "precision") + r"""): the local measurement removes the false
lines of the U-Net. A narrower strip loses recall; a wider one
(""" + v("B6", "f1") + r""" at $\pm$6 px) gains little.

\textbf{Rows 7--8: the references.} At the same cost as row 5, a raster
reaches only """ + v("CB5", "f1") + r""". Measuring everything reaches
""" + v("F", "f1") + r""", """ + vs + r""" row 5 (""" + v("B5", "f1") + r""").

\section*{In one sentence}
With the same few measurements the U-Net finds the lines about twice as
well as a raster scan; and when its lines are used to decide where to
measure next, the result is """ + vs + r""" a full scan, from
""" + m("B5") + r"""\,\% of the measurements.

\section*{Limits}
\begin{itemize}
\item Simulated, noise-free devices; noise will affect every row.
\item The strip is counted pixel by pixel. Short sweeps across each line
  would cost less, so the """ + m("B5") + r"""\,\% overstates the cost of
  the second step.
\item A learned reconstruction of the sparse measurement may do better than
  a raster and has not been compared.
\item Lines the U-Net misses entirely cannot be recovered by the local
  check, because it measures only near the predicted lines.
\end{itemize}
\end{document}
"""
    os.makedirs(BUILD, exist_ok=True)
    with open(os.path.join(BUILD, "Table_III_explained.tex"), "w",
              encoding="utf-8") as f:
        f.write(tex)
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode",
                        "Table_III_explained.tex"], cwd=BUILD,
                       capture_output=True)
    pdf = os.path.join(BUILD, "Table_III_explained.pdf")
    os.replace(pdf, os.path.join(OUT, "Table_III_explained.pdf"))
    print("wrote", os.path.join(OUT, "Table_III_explained.pdf"))


if __name__ == "__main__":
    main()
