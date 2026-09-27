/*
 * build_overview.jsx -- Figure 1 of the manuscript, drawn in Illustrator.
 *
 * THE STYLE IS THE SSDM SLIDE'S and is kept deliberately: two large
 * overlapping discs in the same two teals, the U-Net as a pale pill in the
 * overlap, rounded callout boxes with white bold text around the outside,
 * a dashed schematic in the simulated disc and a solid one in the measured
 * disc.  What changed is the CONTENT of those boxes and two small figures
 * added, which is what the review actually needed:
 *
 *   - the right disc no longer claims a result.  It says "not
 *     demonstrated in this work" in the disc itself, and the only box
 *     attached to it calls it the intended application;
 *   - the rays are in the figure, over a stability diagram, so the method
 *     cannot be read as a grid mask, and the two channels and the
 *     probability map are there too;
 *   - "Digital twin" is gone: an ensemble of randomly drawn capacitances
 *     is not a twin of any one device.  "Extract transition lines" is gone
 *     with it: they are computed from the charge configuration, not found
 *     in an image;
 *   - the imperative, half-capitalised box text ("Change the Capacitances
 *     of the model", "Perform Sparse measurement") is gone; the boxes are
 *     noun phrases in sentence case;
 *   - the dashed-versus-solid schematic is explained in words.
 *
 * The DATA is not drawn here.  tile_*.png are written from the run by
 * make_figures.fig_overview_tiles() and placed.
 */
#target illustrator

function cm(v) { return v * 28.3464567; }

var HERE = new File($.fileName).parent;
var TILES = new Folder(HERE.parent.parent +
    "/paper_figures/method_docs/2_Introduction/figures");
var OUT = HERE;

var W = cm(17.8), H = cm(9.60);
var doc = app.documents.add(DocumentColorSpace.RGB, W, H);
doc.rulerOrigin = [0, 0];

function rgb(r, g, b) {
    var c = new RGBColor(); c.red = r; c.green = g; c.blue = b; return c;
}
// the slide's own palette
var DISC_L  = rgb(61, 143, 166);      // simulated
var DISC_R  = rgb(79, 185, 172);      // measured
var BOX_L   = rgb(46, 111, 142);
var BOX_R   = rgb(69, 179, 163);
var PILL    = rgb(232, 226, 214);
var INK     = rgb(26, 26, 26);
var WHITE   = rgb(255, 255, 255);
var FRAME   = rgb(255, 255, 255);

function font(bold) {
    try { return app.textFonts.getByName(bold ? "Arial-BoldMT" : "ArialMT"); }
    catch (e) { return app.textFonts[0]; }
}

/* (x, yTop) is where the bounding box's centre-top goes when align is
 * "c" -- the only way a multi-line label lands where the layout says. */
function label(text, x, yTop, size, colour, bold, align) {
    var t = doc.textFrames.add();
    t.contents = text;
    var a = t.textRange.characterAttributes;
    a.size = size; a.fillColor = colour; a.textFont = font(bold);
    a.leading = size * 1.24;
    t.textRange.paragraphAttributes.justification =
        (align === "c") ? Justification.CENTER : Justification.LEFT;
    t.top = yTop;
    t.left = (align === "c") ? x - t.width / 2 : x;
    return t;
}

/* A slide-style callout: rounded box, white bold text, sized to the text. */
function callout(text, cx, cy, size, fill) {
    var t = label(text, cx, cy, size, WHITE, true, "c");
    var padX = cm(0.30), padY = cm(0.22);
    var w = t.width + 2 * padX, h = t.height + 2 * padY;
    var r = doc.pathItems.roundedRectangle(cy + padY, cx - w / 2, w, h,
                                           cm(0.22), cm(0.22));
    r.filled = true; r.fillColor = fill; r.stroked = false;
    r.zOrder(ZOrderMethod.SENDTOBACK);
    t.zOrder(ZOrderMethod.BRINGTOFRONT);
    return r;
}

function disc(cx, cy, r, fill) {
    var e = doc.pathItems.ellipse(cy + r, cx - r, 2 * r, 2 * r);
    e.filled = true; e.fillColor = fill; e.stroked = false;
    e.blendingMode = BlendModes.MULTIPLY;   // the overlap darkens, as on
    e.opacity = 88;                         // the slide
    return e;
}

function place(name, cx, cy, w) {
    var f = new File(TILES + "/" + name);
    if (!f.exists) { throw new Error("missing tile: " + f.fsName); }
    var p = doc.placedItems.add();
    p.file = f; p.embed();
    var art = doc.pageItems[0];
    var k = w / art.width;
    art.width *= k; art.height *= k;
    art.left = cx - art.width / 2;
    art.top = cy + art.height / 2;
    var fr = doc.pathItems.rectangle(art.top, art.left, art.width,
                                     art.height);
    fr.filled = false; fr.stroked = true;
    fr.strokeColor = FRAME; fr.strokeWidth = 0.9;
    return art;
}

function line(x1, y1, x2, y2, colour, wid, dashed) {
    var p = doc.pathItems.add();
    p.setEntirePath([[x1, y1], [x2, y2]]);
    p.filled = false; p.stroked = true;
    p.strokeColor = colour; p.strokeWidth = wid;
    if (dashed) { p.strokeDashes = [4, 3]; }
    return p;
}

function arrow(x1, y1, x2, y2, colour, wid, dashed) {
    line(x1, y1, x2, y2, colour, wid, dashed);
    var ang = Math.atan2(y2 - y1, x2 - x1), s = cm(0.20);
    var h = doc.pathItems.add();
    h.setEntirePath([
        [x2, y2],
        [x2 - s * Math.cos(ang - 0.38), y2 - s * Math.sin(ang - 0.38)],
        [x2 - s * Math.cos(ang + 0.38), y2 - s * Math.sin(ang + 0.38)]
    ]);
    h.closed = true; h.filled = true; h.fillColor = colour; h.stroked = false;
}

/* The double-dot schematic of the slide: three gates over a channel, two
 * dots on it and a sensor hanging off it.  `dashed` draws the simulated
 * one, `solid` the fabricated one -- the distinction the review asked to
 * have stated, and it is stated in the boxes. */
function device(cx, cy, s, dashed, stroke, txt) {
    var gw = s * 0.16, gh = s * 0.40, gy = cy + s * 0.30;
    for (var i = -1; i <= 1; i++) {
        var g = doc.pathItems.roundedRectangle(gy + gh / 2,
                                               cx + i * s * 0.28 - gw / 2,
                                               gw, gh, gw * 0.4, gw * 0.4);
        g.filled = false; g.stroked = true;
        g.strokeColor = stroke; g.strokeWidth = s * 0.030;
        g.strokeDashes = dashed ? [s * 0.07, s * 0.05] : [];
    }
    var ly = cy - s * 0.02;
    var ch = line(cx - s * 0.52, ly, cx + s * 0.52, ly, stroke, s * 0.030,
                  dashed);
    if (!dashed) { ch.strokeDashes = []; }
    for (var j = -1; j <= 1; j += 2) {
        var d = doc.pathItems.ellipse(ly + s * 0.11, cx + j * s * 0.15 -
                                      s * 0.11, s * 0.22, s * 0.22);
        d.filled = true; d.fillColor = WHITE;
        d.stroked = true; d.strokeColor = stroke; d.strokeWidth = s * 0.030;
        d.strokeDashes = dashed ? [s * 0.07, s * 0.05] : [];
        label(j < 0 ? "1" : "2", cx + j * s * 0.15, ly + s * 0.085,
              s * 0.115, stroke, true, "c");
    }
    var st = line(cx + s * 0.40, ly, cx + s * 0.40, ly - s * 0.24, stroke,
                  s * 0.030, dashed);
    if (!dashed) { st.strokeDashes = []; }
    var sd = doc.pathItems.ellipse(ly - s * 0.24, cx + s * 0.40 - s * 0.10,
                                   s * 0.20, s * 0.20);
    sd.filled = true; sd.fillColor = WHITE;
    sd.stroked = true; sd.strokeColor = stroke; sd.strokeWidth = s * 0.030;
    sd.strokeDashes = dashed ? [s * 0.07, s * 0.05] : [];
    label("sensor", cx + s * 0.40, ly - s * 0.48, s * 0.10, stroke, true,
          "c");
    if (txt) { label(txt, cx, gy + gh / 2 + s * 0.14, s * 0.10, stroke,
                     false, "c"); }
}

// ---- the two discs ------------------------------------------------------
var CY = cm(5.05), RD = cm(3.05);
var CXL = cm(5.15), CXR = cm(9.65);
disc(CXL, CY, RD, DISC_L);
disc(CXR, CY, RD, DISC_R);

// ---- left disc: the simulator ------------------------------------------
device(CXL - cm(0.30), CY + cm(1.30), cm(2.20), true, WHITE,
       "capacitances varied");
label("Simulated devices", CXL - cm(0.55), CY - cm(1.05), 12, WHITE,
      true, "c");
label("constant-capacitance simulator", CXL - cm(0.55), CY - cm(1.58),
      7.5, WHITE,
      false, "c");

// ---- right disc: the device this work does not have ---------------------
device(CXR + cm(0.25), CY + cm(1.48), cm(2.20), false, INK, null);
label("Measured device", CXR + cm(0.70), CY - cm(1.05), 12, INK, true,
      "c");
label("a fabricated double dot in a dilution refrigerator",
      CXR + cm(0.70),
      CY - cm(1.62), 7.0, INK, false, "c");
label("not demonstrated in this work", CXR + cm(0.70), CY - cm(2.10),
      8, INK, true,
      "c");

// ---- the overlap: the network ------------------------------------------
var UX = (CXL + RD + CXR - RD) / 2;
var pill = doc.pathItems.roundedRectangle(CY + cm(0.34), UX - cm(0.92),
                                          cm(1.84), cm(0.68),
                                          cm(0.18), cm(0.18));
pill.filled = true; pill.fillColor = PILL; pill.stroked = false;
label("U-Net", UX, CY + cm(0.20), 10, rgb(26, 70, 86), true, "c");

// ---- the callouts, and the small figures they name ----------------------
// left: what the simulator supplies
callout("capacitances drawn at random\nfor each simulated device",
        cm(2.55), cm(8.95), 7.2, BOX_L);
callout("exact transition lines from\nthe charge configuration",
        cm(1.95), cm(5.60), 7.2, BOX_L);
callout("the network is trained\non the simulated diagrams",
        cm(2.65), cm(1.85), 7.2, BOX_L);

// right: the measurement, the input, the output -- all of it on simulated
// devices, which is why these boxes hang off the CENTRE, not off the
// measured disc
callout("sparse measurement:\na fan of rays from one corner",
        cm(12.30), cm(9.15), 7.2, BOX_R);
callout("the two input channels:\nsampled signal, sampled mask",
        cm(12.30), cm(4.95), 7.2, BOX_R);
callout("probability map of the\ntransition lines", cm(12.30), cm(2.05),
        7.2, BOX_R);

var TW = cm(1.50);
place("tile_rays.png", cm(15.85), cm(8.62), TW);
place("tile_channels.png", cm(15.85), cm(4.42), cm(2.00));
place("tile_probability.png", cm(15.85), cm(1.52), TW);

// the one dashed arrow: the step this paper does not take
arrow(UX, CY - cm(0.44), CXR + cm(0.20), CY - cm(0.44), INK, 1.1, true);
label("intended application", (UX + CXR + cm(0.20)) / 2, CY - cm(0.54), 7,
      INK, false, "c");

// ---- the page is trimmed to the drawing --------------------------------
var vb = doc.visibleBounds;
var m = cm(0.18);
doc.artboards[0].artboardRect = [vb[0] - m, vb[1] + m, vb[2] + m, vb[3] - m];

// ---- out ----------------------------------------------------------------
var opt = new ExportOptionsPNG24();
opt.antiAliasing = true; opt.transparency = false;
opt.artBoardClipping = true;
opt.horizontalScale = 600 / 72 * 100;
opt.verticalScale = 600 / 72 * 100;
doc.exportFile(new File(OUT + "/method_overview.png"), ExportType.PNG24, opt);

var pdfOpt = new PDFSaveOptions();
pdfOpt.compatibility = PDFCompatibility.ACROBAT5;
pdfOpt.preserveEditability = false;
doc.saveAs(new File(OUT + "/method_overview.pdf"), pdfOpt);
doc.saveAs(new File(OUT + "/method_overview.ai"),
           new IllustratorSaveOptions());
doc.close(SaveOptions.DONOTSAVECHANGES);
"ok";
