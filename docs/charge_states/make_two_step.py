# -*- coding: utf-8 -*-
"""
make_two_step.py -- a short PDF: U-Net first, then local check.

    python scripts/run_19_two_step.py
    python docs/charge_states/make_two_step.py     # writes and compiles

Every number is read from results/<run>/two_step/table.csv.
"""
import csv
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUN = "4-5-6-7-8_rays_40-50-60_points_500_samples"
OUT = os.path.join(ROOT, "results", RUN, "two_step")


def main():
    with open(os.path.join(OUT, "table.csv"), newline="",
              encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    g = lambda m, s="": next(r for r in rows if r["method"] == m  # noqa
                             and r["setting"].startswith(s))
    u, full = g("U-Net only"), g("full scan")
    two = g("U-Net, then local check", "band ±5")
    c5 = g("coarse scan", "22 x 22")
    c59 = g("coarse scan", "77 x 77")
    f = lambda r, k="f1": f"{float(r[k]):.3f}"                    # noqa
    m = lambda r: f"{float(r['measured']):.1f}"                   # noqa
    body = []
    for r in rows:
        body.append(" & ".join([r["method"], r["setting"].replace("±", "$\\pm$"),
                                m(r), f(r, "precision"), f(r, "recall"),
                                f(r)]) + r" \\")
        if r["method"] == "U-Net only" or r["setting"].startswith("band ±5") \
                or r["method"] == "full scan":
            body.append(r"\midrule")
    tex = r"""\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{lmodern}
\usepackage[margin=2.2cm]{geometry}\usepackage{graphicx}\usepackage{booktabs}
\usepackage{caption}\usepackage{float}\usepackage{parskip}
\captionsetup{font=small,labelfont=bf}
\begin{document}
\begin{center}{\Large U-Net first, then a local check}\\[4pt]
\small 8 rays $\times$ 60 points, 50 test devices,
\texttt{scripts/run\_19\_two\_step.py}\end{center}

\textbf{Idea.} Step 1: measure a few rays; the U-Net says where the
transition lines are. Step 2: measure only a narrow band around those lines
and keep the lines where the signal really steps.

\textbf{Result.} The U-Net alone, from """ + m(u) + r"""\,\% of the plane,
finds the lines with F1@1 = """ + f(u) + r""". A coarse scan with the same
number of points reaches only """ + f(c5) + r""". Adding the local check
in a $\pm$5\,px band (""" + m(two) + r"""\,\% measured) gives
F1@1 = """ + f(two) + r""", better than scanning the whole plane
(""" + f(full) + r""") and better than a coarse scan of the same cost
(""" + f(c59) + r""").

\begin{table}[H]\centering
\caption{Line accuracy on the 50 test devices. The detector threshold of
every row is chosen on the 75 validation devices.}
\small
\begin{tabular}{llrrrr}\toprule
method & setting & measured (\%) & precision & recall & F1@1 \\\midrule
""" + "\n".join(body[:-1] if body[-1] == r"\midrule" else body) + r"""
\bottomrule\end{tabular}\end{table}

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{fig_example.png}
\caption{One test device. (a) The rays. (b) The U-Net lines. (c) The band
measured in step 2. (d) Lines found (red) on the true lines (grey).}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=0.6\textwidth]{fig_f1_vs_measured.png}
\caption{Line accuracy against the fraction of the plane measured.}
\end{figure}

\textbf{Limits.} Noise-free simulations; the band is counted pixel by
pixel (short sweeps across each line would cost less); this measures the
lines, not the charge states.
\end{document}
"""
    path = os.path.join(OUT, "two_step_results.tex")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(tex)
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode",
                        "two_step_results.tex"], cwd=OUT,
                       capture_output=True)
    print("wrote", path.replace(".tex", ".pdf"))


if __name__ == "__main__":
    main()
