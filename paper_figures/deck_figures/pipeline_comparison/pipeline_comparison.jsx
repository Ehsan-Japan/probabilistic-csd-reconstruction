var doc = app.documents.add(DocumentColorSpace.RGB, 720, 306);

var ab = doc.artboards[0].artboardRect;
function X(x){ return ab[0] + x; }
function Y(y){ return ab[1] - y; }
function col(c){ var k = new RGBColor(); k.red=c[0]; k.green=c[1]; k.blue=c[2]; return k; }
function rect(x, y, w, h, fill, stroke, sw){
  var r = doc.pathItems.rectangle(Y(y), X(x), w, h);
  if (fill) { r.filled = true; r.fillColor = col(fill); } else r.filled = false;
  if (stroke) { r.stroked = true; r.strokeColor = col(stroke); r.strokeWidth = sw; } else r.stroked = false;
  return r;
}
function line(pts, stroke, sw, dash){
  var p = doc.pathItems.add(), q = [];
  for (var k = 0; k < pts.length; k++) q.push([X(pts[k][0]), Y(pts[k][1])]);
  p.setEntirePath(q); p.filled = false; p.stroked = true;
  p.strokeColor = col(stroke); p.strokeWidth = sw;
  if (dash) p.strokeDashes = dash;
  return p;
}
function dot(x, y, r, fill){
  var e = doc.pathItems.ellipse(Y(y - r), X(x - r), 2*r, 2*r);
  e.filled = true; e.fillColor = col(fill); e.stroked = false; return e;
}
function head(x, y, dx, dy, s, fill){
  // filled arrowhead with its tip at (x, y), pointing along (dx, dy)
  var n = Math.sqrt(dx*dx + dy*dy); dx /= n; dy /= n;
  var bx = x - dx*s, by = y - dy*s, px_ = -dy*s*0.42, py_ = dx*s*0.42;
  var p = doc.pathItems.add();
  p.setEntirePath([[X(x), Y(y)], [X(bx+px_), Y(by+py_)], [X(bx-px_), Y(by-py_)]]);
  p.closed = true; p.filled = true; p.fillColor = col(fill); p.stroked = false;
}
function arrow(x0, y0, x1, y1, c, sw, s){
  line([[x0, y0], [x1 - (x1-x0)*0.0, y1]], c, sw);
  head(x1, y1, x1 - x0, y1 - y0, s, c);
}
// rich text: segments [[text, font, subscript?], ...]
function text(x, y, segs, size, c, just){
  var t = doc.textFrames.add();
  var s = ""; for (var k = 0; k < segs.length; k++) s += segs[k][0];
  t.contents = s;
  var ca = t.textRange.characterAttributes;
  ca.size = size; ca.fillColor = col(c);
  var at = 0;
  for (var k = 0; k < segs.length; k++) {
    var L = segs[k][0].length;
    for (var m = 0; m < L; m++) {
      var ch = t.textRange.characters[at + m].characterAttributes;
      ch.textFont = app.textFonts.getByName(segs[k][1]);
      if (segs[k][2]) ch.baselinePosition = FontBaselineOption.SUBSCRIPT;
    }
    at += L;
  }
  if (just == "c") t.textRange.paragraphAttributes.justification = Justification.CENTER;
  if (just == "r") t.textRange.paragraphAttributes.justification = Justification.RIGHT;
  t.left = X(x); t.top = Y(y);
  if (just == "c") t.left = X(x) - t.width/2;
  if (just == "r") t.left = X(x) - t.width;
  return t;
}
function boxed(t){
  var g = t.geometricBounds, pd = 2;
  var r = doc.pathItems.rectangle(g[1] + pd, g[0] - pd, g[2] - g[0] + 2*pd, g[1] - g[3] + 2*pd);
  r.filled = true; r.fillColor = col([255,255,255]); r.stroked = false; r.opacity = 85;
  t.zOrder(ZOrderMethod.BRINGTOFRONT);
}
var R = "ArialMT", I = "Arial-ItalicMT", B = "Arial-BoldMT";


function place(path, x, y, s){
  var p = doc.placedItems.add(); p.file = new File(path);
  p.width = s; p.height = s; p.left = X(x); p.top = Y(y); p.embed();
}
function box(x, y, w, h, stroke, fill){
  var r = doc.pathItems.roundedRectangle(Y(y), X(x), w, h, 5, 5);
  r.filled = true; r.fillColor = col(fill); r.stroked = true;
  r.strokeColor = col(stroke); r.strokeWidth = 1.0; return r;
}

var hb = rect(22,16,11,11,[255,255,255],[34,34,34],0.6);
text(27.5,17.3,[["a",B]],8,[34,34,34],"c");
text(39,17,[["Hernandes et al. [12]:",B],[" reconstruct the image, then find the lines",R]],9,[122,122,122],"l");
place("F:/Seminar Fujita/projects/probabilistic-csd-reconstruction/paper_figures/deck_figures/pipeline_comparison/tiles/input_cuts.png",22,40,78);
rect(22,40,78,78,null,[102,102,102],0.5);
text(61,123,[["sparse data:",R]],7.5,[102,102,102],"c");
text(61,132.5,[["line cuts",R]],7.5,[102,102,102],"c");
arrow(106,79,120,79,[122,122,122],1.0,6);
box(126,40,104,78,[122,122,122],[242,242,242]);
text(178,57.4,[["conditional",B]],8,[34,34,34],"c");
text(178,68.2,[["diffusion model",B]],8,[34,34,34],"c");
text(178,79,[["loss: MSE to the",R]],8,[34,34,34],"c");
text(178,89.8,[["measured image",R]],8,[34,34,34],"c");
arrow(236,79,250,79,[122,122,122],1.0,6);
place("F:/Seminar Fujita/projects/probabilistic-csd-reconstruction/paper_figures/deck_figures/pipeline_comparison/tiles/sensor.png",256,40,78);
rect(256,40,78,78,null,[102,102,102],0.5);
text(295,123,[["full charge-sensor",R]],7.5,[34,34,34],"c");
text(295,132.5,[["image, 128 \u00D7 128",R]],7.5,[34,34,34],"c");
arrow(340,79,354,79,[122,122,122],1.0,6);
box(360,40,104,78,[122,122,122],[242,242,242]);
text(412,57.4,[["Canny edges",B]],8,[34,34,34],"c");
text(412,68.2,[["+ Frangi ridges",B]],8,[34,34,34],"c");
text(412,79,[["classical filters,",R]],8,[34,34,34],"c");
text(412,89.8,[["after the network",R]],8,[34,34,34],"c");
arrow(470,79,484,79,[122,122,122],1.0,6);
place("F:/Seminar Fujita/projects/probabilistic-csd-reconstruction/paper_figures/deck_figures/pipeline_comparison/tiles/lines_truth.png",490,40,78);
rect(490,40,78,78,null,[102,102,102],0.5);
text(529,123,[["extracted lines",R]],7.5,[102,102,102],"c");
text(590,58,[["evaluated on",R]],7.5,[102,102,102],"l");
text(590,70,[["image: ",B],["rNMSE, PSNR, SSIM",R]],7.5,[34,34,34],"l");
text(590,81,[["lines: ",B],["IoU, F1, Hausdorff",R]],7.5,[34,34,34],"l");
var hb = rect(22,162,11,11,[255,255,255],[34,34,34],0.6);
text(27.5,163.3,[["b",B]],8,[34,34,34],"c");
text(39,163,[["This work:",B],[" infer the lines directly",R]],9,[8,81,156],"l");
place("F:/Seminar Fujita/projects/probabilistic-csd-reconstruction/paper_figures/deck_figures/pipeline_comparison/tiles/input.png",22,186,78);
rect(22,186,78,78,null,[102,102,102],0.5);
text(61,269,[["sparse data:",R]],7.5,[102,102,102],"c");
text(61,278.5,[["rays",R]],7.5,[102,102,102],"c");
arrow(106,225,120,225,[8,81,156],1.0,6);
box(126,186,104,78,[8,81,156],[234,242,250]);
text(178,203.4,[["U-Net",B]],8,[34,34,34],"c");
text(178,214.2,[["loss: weighted BCE",R]],8,[34,34,34],"c");
text(178,225,[["+ Dice to the",R]],8,[34,34,34],"c");
text(178,235.8,[["line labels",R]],8,[34,34,34],"c");
arrow(236,225,250,225,[8,81,156],1.0,6);
place("F:/Seminar Fujita/projects/probabilistic-csd-reconstruction/paper_figures/deck_figures/pipeline_comparison/tiles/probability.png",256,186,78);
rect(256,186,78,78,null,[102,102,102],0.5);
text(295,269,[["P(transition line)",R]],7.5,[34,34,34],"c");
text(295,278.5,[["per pixel, 100 \u00D7 100",R]],7.5,[34,34,34],"c");
arrow(340,225,354,225,[8,81,156],1.0,6);
box(360,186,104,78,[8,81,156],[234,242,250]);
text(412,208.8,[["threshold",B]],8,[34,34,34],"c");
text(412,219.6,[["chosen on",R]],8,[34,34,34],"c");
text(412,230.4,[["validation devices",R]],8,[34,34,34],"c");
arrow(470,225,484,225,[8,81,156],1.0,6);
place("F:/Seminar Fujita/projects/probabilistic-csd-reconstruction/paper_figures/deck_figures/pipeline_comparison/tiles/lines_pred.png",490,186,78);
rect(490,186,78,78,null,[102,102,102],0.5);
text(529,269,[["predicted lines",R]],7.5,[102,102,102],"c");
text(590,204,[["evaluated on",R]],7.5,[102,102,102],"l");
text(590,216,[["lines: ",B],["F1@\u03C4",R]],7.5,[34,34,34],"l");
var hl = rect(251,181,88,122,null,[8,81,156],1.2); hl.strokeDashes=[3,2];
var hl2 = rect(251,35,88,122,null,[122,122,122],1.2); hl2.strokeDashes=[3,2];
text(295,145,[["network output",B]],7.5,[122,122,122],"c");
text(295,291,[["network output",B]],7.5,[8,81,156],"c");

var base = "F:/Seminar Fujita/projects/probabilistic-csd-reconstruction/paper_figures/deck_figures/pipeline_comparison/pipeline_comparison";
var o = new IllustratorSaveOptions(); o.pdfCompatible = true;
doc.saveAs(new File(base + ".ai"), o);
var pdf = new PDFSaveOptions(); pdf.preserveEditability = false;
doc.saveAs(new File(base + ".pdf"), pdf);
var png = new ExportOptionsPNG24(); png.horizontalScale = 400; png.verticalScale = 400;
png.antiAliasing = true; png.transparency = false; png.artBoardClipping = true;
doc.exportFile(new File(base + ".png"), ExportType.PNG24, png);
var svg = new ExportOptionsSVG(); svg.embedRasterImages = true;
svg.fontType = SVGFontType.OUTLINEFONT;
doc.exportFile(new File(base + ".svg"), ExportType.SVG, svg);
doc.close(SaveOptions.DONOTSAVECHANGES);
"done";
