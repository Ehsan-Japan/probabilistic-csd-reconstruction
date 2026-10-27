"""
make_fig6_unet3d.py -- Fig. 6 of Manuscript_ver6: the U-Net drawn as stacks
of feature-map slabs (in the style of the user's reference,
method_docs_ver2/unet_sample.png; user, 2026-10-27).

    python paper_figures/programs/make_fig6_unet3d.py

The structure is read from the network itself (csdrecon.ml.grid_model):
every encoder stage is two 3x3 convolutions (each with batch norm and GELU)
followed by 2x2 max pooling, the bottleneck two convolutions, every decoder
stage a nearest-neighbour upsampling, the concatenation with the matching
encoder output and two convolutions, and the head a 1x1 convolution with a
sigmoid.  Widths and grid sizes come from the model: 32, 64, 128 / 256 on
100, 50, 25 / 12 pixels.  The input planes are the two channels of the
picked test device and the output plane its probability map
(make_figures.pick_case), as in the previous Fig. 6.  Slab height follows
the grid size and slab thickness the channel count, both schematically.
Writes method_docs_ver2/3_methods/figures/fig_model_flow.{svg,png}; the
manuscript generator swaps the PNG in for Fig. 6.
"""
import base64
import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.colors import LinearSegmentedColormap      # noqa: E402

import make_figures as MF                                  # noqa: E402
from csdrecon.ml import ray_peaks                          # noqa: E402
from csdrecon.ml.grid_model import RayToLinesNet, WIDTH    # noqa: E402

OUT = Path(MF.ROOT) / "paper_figures" / "method_docs_ver2" / "3_methods" \
    / "figures"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
W = 439.0                                   # pt, the printed width
FONT = "Arial, Helvetica, sans-serif"
INK, MUTED = "#111827", "#4b5563"
# (face, front) colours, after the reference figure
CONV = ("#4a7ebb", "#6a9fd8")
POOL = ("#548235", "#70ad47")
UP = ("#e0a100", "#ffc000")
HEAD = ("#a0aec0", "#cbd5e0")
SKIP = "#5b9bd5"


def uri(a, cmap, vmin=0.0, vmax=1.0):
    buf = io.BytesIO()
    plt.imsave(buf, a, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax,
               format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def height(s):                  # slab height from the grid size (pixels)
    return 24 + 0.66 * s


def thick(c):                   # slab thickness from the channel count
    return 2.4 + 1.1 * np.log2(max(c, 1) / WIDTH + 1)


class Svg:
    def __init__(self):
        self.el = []

    def add(self, s):
        self.el.append(s)

    def text(self, x, y, s, size=7.5, fill=INK, weight="normal",
             anchor="middle", style="normal"):
        self.add('<text x="%.2f" y="%.2f" font-size="%g" fill="%s" '
                 'font-weight="%s" font-style="%s" text-anchor="%s">%s'
                 '</text>' % (x, y, size, fill, weight, style, anchor,
                              s.replace("&", "&amp;")))

    def poly(self, pts, fill, stroke="#ffffff", sw=0.35):
        self.add('<polygon points="%s" fill="%s" stroke="%s" '
                 'stroke-width="%g" stroke-linejoin="round"/>'
                 % (" ".join("%.2f,%.2f" % p for p in pts), fill, stroke, sw))

    def path(self, pts, stroke, sw):
        self.add('<path d="M%s" fill="none" stroke="%s" stroke-width="%g" '
                 'stroke-linejoin="round"/>'
                 % (" L".join("%.2f %.2f" % p for p in pts), stroke, sw))

    def head(self, x, y, colour, s=3.6):
        """an arrowhead pointing down, its tip at (x, y)"""
        self.poly([(x, y), (x - s * 0.55, y - s), (x + s * 0.55, y - s)],
                  colour, colour, 0.1)

    def render(self, h, y0):
        return ('<svg xmlns="http://www.w3.org/2000/svg" width="%gpt" '
                'height="%gpt" viewBox="0 %.2f %g %g" font-family="%s">\n'
                '<rect x="0" y="%.2f" width="%g" height="%g" fill="#ffffff"/>'
                '\n%s\n</svg>\n' % (W, h, y0, W, h, FONT, y0, W, h,
                                    "\n".join(self.el)))


def build(sig_uri, mask_uri, prob_uri, net):
    S = Svg()
    yc = 0.0                                   # the slabs' centre line
    widths = net.widths                        # 32, 64, 128, 256
    sizes = [100]
    for _ in range(net.depth):
        sizes.append(sizes[-1] // 2)           # 100, 50, 25, 12

    def slab(x, s, c, colours):
        """one feature map: a thin front strip and its face, drawn in
        oblique projection; returns the face's far top corner"""
        h, t = height(s), thick(c)
        d = 0.5 * h
        top, bot = yc - h / 2, yc + h / 2
        S.poly([(x + t, top), (x + t + d, top - d), (x + t + d, bot - d),
                (x + t, bot)], colours[0])
        S.poly([(x, top), (x + t, top), (x + t + d, top - d),
                (x + d, top - d)], colours[1])
        S.poly([(x, top), (x + t, top), (x + t, bot), (x, bot)], colours[1])
        return (x + t + d, top - d), x + t

    def plane(x, s, href):
        """an input or output map, drawn on a slanted plane"""
        h = height(s)
        d = 0.5 * h
        top = yc - h / 2
        S.add('<image x="0" y="0" width="%.2f" height="%.2f" '
              'preserveAspectRatio="none" transform="matrix(1 -1 0 1 %.2f '
              '%.2f)" href="%s"/>' % (d, h, x, top, href))
        S.poly([(x, top), (x + d, top - d), (x + d, top - d + h),
                (x, top + h)], "none", "#374151", 0.6)

    groups, skips = [], {}
    # the input: channel 2 (mask) behind channel 1 (signal)
    x_in = 6.0
    plane(x_in, 100, mask_uri)
    plane(x_in + 7, 100, sig_uri)
    in_right = x_in + 7 + 0.5 * height(100)
    box_l = in_right + 15                      # the dashed U-Net box
    x = box_l + 5
    pitch = 31.0
    # encoder
    for k, (s, c) in enumerate(zip(sizes[:-1], widths[:-1])):
        x0 = x
        for _ in range(2):
            corner, x = slab(x, s, c, CONV)
            x += 1.0
        skips[k] = corner
        _c, x = slab(x, s, c, POOL)
        groups.append((x0, s, c))
        x = x0 + pitch
    # bottleneck
    x0 = x
    for _ in range(2):
        _c, x = slab(x, sizes[-1], widths[-1], CONV)
        x += 1.0
    groups.append((x0, sizes[-1], widths[-1]))
    x = x0 + pitch * 0.85
    # decoder
    lands = {}
    for k in reversed(range(net.depth)):
        s, c = sizes[k], widths[k]
        x0 = x
        _c, x = slab(x, s, widths[k + 1], UP)
        x += 1.0
        corner, x = slab(x, s, c, CONV)
        lands[k] = corner
        x += 1.0
        _c, x = slab(x, s, c, CONV)
        groups.append((x0, s, c))
        x = x0 + pitch if k else x + 7
    # head: 1x1 convolution and sigmoid
    _c, x = slab(x, 100, 1, HEAD)
    x_head = x
    box_r = x_head + 0.5 * height(100) + 5
    # the output
    x_out = box_r + 15
    plane(x_out, 100, prob_uri)

    # concatenation: from each encoder stage to its decoder stage
    top_all = yc - height(100) / 2 - 0.5 * height(100)
    for k in range(net.depth):
        (xs, ys), (xe, ye) = skips[k], lands[k]
        yl = top_all - 8 - 9 * (net.depth - 1 - k)
        S.path([(xs, ys), (xs, yl), (xe, yl), (xe, ye - 3.6)], SKIP, 1.0)
        S.head(xe, ye, SKIP)
    y_conc = top_all - 8 - 9 * (net.depth - 1) - 4
    S.text((skips[0][0] + lands[0][0]) / 2, y_conc, "Concatenate", 7.5, INK)

    # sizes and widths under every stage
    for x0, s, c in groups:
        bot = yc + height(s) / 2
        S.text(x0 + 2, bot + 9, "%d × %d" % (s, s), 7, INK,
               anchor="start")
        S.text(x0 + 2, bot + 17, "%d ch" % c, 6.5, MUTED, anchor="start")
    bot = yc + height(100) / 2

    # the dashed box around the network, named above it (user, 2026-10-27)
    box_t, box_b = y_conc - 11, bot + 22
    S.add('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" rx="5" '
          'fill="none" stroke="#374151" stroke-width="0.9" '
          'stroke-dasharray="4 2.5"/>' % (box_l, box_t, box_r - box_l,
                                          box_b - box_t))
    S.text((box_l + box_r) / 2, box_t - 4, "U-Net", 9, INK, "bold")
    # arrows: input -> U-Net -> output, at the planes' middle height
    ya = yc - 0.25 * height(100)
    for x0, x1 in ((in_right + 2, box_l - 2), (box_r + 2, x_out - 2)):
        S.path([(x0, ya), (x1 - 5, ya)], INK, 1.3)
        S.add('<polygon points="%.2f,%.2f %.2f,%.2f %.2f,%.2f" fill="%s"/>'
              % (x1, ya, x1 - 5.5, ya - 2.6, x1 - 5.5, ya + 2.6, INK))

    S.text(x_in + 22, bot + 9, "Input", 7, INK)
    S.text(x_in + 22, bot + 17, "2 × 100 × 100", 6.5, MUTED)
    S.text(x_in + 22, bot + 25, "signal, mask", 6.5, MUTED)
    S.text(x_out + 22, bot + 9, "Output", 7, INK)
    S.text(x_out + 22, bot + 17, "100 × 100", 6.5, MUTED)
    S.text(x_out + 22, bot + 25, "P(transition line)", 6.5, MUTED)

    # legend, two rows; widths measured with Arial
    from PIL import ImageFont
    font = ImageFont.truetype("arial.ttf", 100)

    def tw(t, size):
        return font.getlength(t) * size / 100.0
    items = [("box", CONV, "3×3 conv + BN + GELU"),
             ("box", POOL, "2×2 max pooling"),
             ("box", UP, "nearest upsampling"),
             ("box", HEAD, "1×1 conv + sigmoid"),
             ("arrow", None, "skip connection (concatenate)")]
    rows = [items[:3], items[3:]]
    ly0 = box_b + 13
    right_leg = 0.0
    for r, row in enumerate(rows):
        ly = ly0 + 11 * r
        tot = sum(18 + tw(lab, 7) for _k, _c, lab in row) + 16 * (len(row) - 1)
        xl = (W - tot) / 2
        for kind, colours, lab in row:
            if kind == "box":
                S.poly([(xl, ly - 6), (xl + 14, ly - 6), (xl + 14, ly + 1),
                        (xl, ly + 1)], colours[1], colours[0], 0.6)
            else:
                S.path([(xl, ly - 2.5), (xl + 10, ly - 2.5)], SKIP, 1.0)
                S.add('<polygon points="%.2f,%.2f %.2f,%.2f %.2f,%.2f" '
                      'fill="%s"/>' % (xl + 14, ly - 2.5, xl + 10, ly - 4.5,
                                       xl + 10, ly - 0.5, SKIP))
            S.text(xl + 18, ly, lab, 7, INK, anchor="start")
            xl += 18 + tw(lab, 7) + 16
            right_leg = max(right_leg, xl - 16)
    ly = ly0 + 11
    right = max(x_out + 0.5 * height(100) + 4, x_out + 22 + tw(
        "P(transition line)", 6.5) / 2, right_leg)
    print("box %.1f-%.1f, output to %.1f, legend to %.1f"
          % (box_l, box_r, right, right_leg))
    assert right <= W - 2, "the figure is %.1f pt wide" % right
    y_top = box_t - 16
    return S.render(ly + 8 - y_top, y_top), ly + 8 - y_top


def main():
    ch, _Yt, p = MF.pick_case()
    sig, vis = ch[ray_peaks.CH_SIGNAL], ch[ray_peaks.CH_VISITED]
    # (user, 2026-10-27) the rays coloured by the measured charge-sensor
    # signal, each widened by one pixel for legibility at print size
    from scipy.ndimage import binary_dilation, grey_dilation
    on = vis > 0.5
    wide = binary_dilation(on)
    val = grey_dilation(np.where(on, sig, -1.0), size=(2, 2))
    col = plt.get_cmap("hot").copy()         # as Fig. 4(a) (user)
    col.set_bad("#f2f2f2")                   # never measured
    sig_uri = uri(np.where(wide & (val >= 0), val, np.nan), col)
    mask_uri = uri(1.0 - vis, "gray")
    blues = LinearSegmentedColormap.from_list("w2b", ["#ffffff", "#2563eb",
                                                      "#1e3a8a"])
    prob_uri = uri(p, "magma")         # as Fig. 7(a) (user)
    net = RayToLinesNet()
    svg, h = build(sig_uri, mask_uri, prob_uri, net)
    OUT.mkdir(parents=True, exist_ok=True)
    svg_p = OUT / "fig_model_flow.svg"
    svg_p.write_text(svg, encoding="utf-8")
    html = OUT / "fig_model_flow.html"
    html.write_text("<!doctype html><html><head><meta charset='utf-8'><style>"
                    "html,body{margin:0;padding:0;background:#fff}svg{display:"
                    "block}</style></head><body>%s</body></html>" % svg,
                    encoding="utf-8")
    wpx, hpx = int(np.ceil(W * 96 / 72)), int(np.ceil(h * 96 / 72))
    subprocess.run([CHROME, "--headless=new", "--disable-gpu",
                    "--hide-scrollbars", "--force-device-scale-factor=6",
                    "--window-size=%d,%d" % (wpx, hpx),
                    "--screenshot=%s" % (OUT / "fig_model_flow.png"),
                    html.resolve().as_uri()], check=True, capture_output=True)
    print("wrote", svg_p, "and fig_model_flow.png (%.0f x %.0f pt)" % (W, h))


if __name__ == "__main__":
    main()
