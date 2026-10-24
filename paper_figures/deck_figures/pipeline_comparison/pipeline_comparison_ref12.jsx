var doc = app.documents.add(DocumentColorSpace.RGB, 684, 388);

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
  var r = doc.pathItems.roundedRectangle(Y(y), X(x), w, h, 6, 6);
  r.filled = true; r.fillColor = col(fill); r.stroked = true;
  r.strokeColor = col(stroke); r.strokeWidth = 1.2; return r;
}

rect(8,4,13,13,[255,255,255],[34,34,34],0.7);
text(14.5,4.8,[["a",B]],10,[34,34,34],"c");
text(28,3.5,[["Hernandes et al. [12]:",B],[" reconstruct the image, then find the lines",R]],13,[110,110,110],"l");
place("F:/Seminar Fujita/projects/probabilistic-csd-reconstruction/paper_figures/deck_figures/pipeline_comparison/tiles/h_masked.png",8,28,112);
rect(8,28,112,112,null,[102,102,102],0.6);
text(64,144,[["sparse data: grid mask,",R]],10,[102,102,102],"c");
text(64,156,[["\u2248 4 % of the image",R]],10,[102,102,102],"c");
arrow(124,84,140,84,[110,110,110],1.3,8);
box(144,28,118,112,[110,110,110],[242,242,242]);
text(203,57.6,[["conditional",B]],11.5,[34,34,34],"c");
text(203,70.8,[["diffusion model",B]],11.5,[34,34,34],"c");
text(203,84,[["loss: MSE to the",R]],10.5,[34,34,34],"c");
text(203,97.2,[["measured image",R]],10.5,[34,34,34],"c");
arrow(266,84,282,84,[110,110,110],1.3,8);
place("F:/Seminar Fujita/projects/probabilistic-csd-reconstruction/paper_figures/deck_figures/pipeline_comparison/tiles/h_diffusion.png",286,28,112);
rect(286,28,112,112,null,[102,102,102],0.6);
text(342,144,[["diffusion reconstruction,",R]],10,[34,34,34],"c");
text(342,156,[["128 \u00D7 128",R]],10,[34,34,34],"c");
text(342,170,[["network output",B]],10,[110,110,110],"c");
var hl = rect(281,23,122,162,null,[110,110,110],1.3); hl.strokeDashes=[3.5,2.5];
arrow(402,84,418,84,[110,110,110],1.3,8);
box(422,28,118,112,[110,110,110],[242,242,242]);
text(481,57.6,[["Canny edges",B]],11.5,[34,34,34],"c");
text(481,70.8,[["+ Frangi ridges",B]],11.5,[34,34,34],"c");
text(481,84,[["applied after",R]],10.5,[34,34,34],"c");
text(481,97.2,[["the network",R]],10.5,[34,34,34],"c");
text(481,144,[["scored on the image",R]],10,[102,102,102],"c");
text(481,156,[["(rNMSE, PSNR, SSIM)",R]],10,[102,102,102],"c");
text(481,168,[["and the lines",R]],10,[102,102,102],"c");
text(481,180,[["(IoU, F1, Hausdorff)",R]],10,[102,102,102],"c");
arrow(544,84,560,84,[110,110,110],1.3,8);
place("F:/Seminar Fujita/projects/probabilistic-csd-reconstruction/paper_figures/deck_figures/pipeline_comparison/tiles/h_ridges.png",564,28,112);
rect(564,28,112,112,null,[102,102,102],0.6);
text(620,144,[["ridges vs measured image:",R]],10,[102,102,102],"c");
text(620,156,[["white both, blue extra,",R]],10,[102,102,102],"c");
text(620,168,[["red missed",R]],10,[102,102,102],"c");
rect(8,200,13,13,[255,255,255],[34,34,34],0.7);
text(14.5,200.8,[["b",B]],10,[34,34,34],"c");
text(28,199.5,[["This work:",B],[" infer the lines directly",R]],13,[8,81,156],"l");
place("F:/Seminar Fujita/projects/probabilistic-csd-reconstruction/paper_figures/deck_figures/pipeline_comparison/tiles/input.png",8,224,112);
rect(8,224,112,112,null,[102,102,102],0.6);
text(64,340,[["sparse data: 8 rays,",R]],10,[102,102,102],"c");
text(64,352,[["4.7 % of the plane",R]],10,[102,102,102],"c");
arrow(124,280,140,280,[8,81,156],1.3,8);
box(144,224,118,112,[8,81,156],[234,242,250]);
text(203,253.6,[["U-Net",B]],11.5,[34,34,34],"c");
text(203,266.8,[["loss: weighted BCE",R]],10.5,[34,34,34],"c");
text(203,280,[["+ Dice to the",R]],10.5,[34,34,34],"c");
text(203,293.2,[["line labels",R]],10.5,[34,34,34],"c");
arrow(266,280,282,280,[8,81,156],1.3,8);
place("F:/Seminar Fujita/projects/probabilistic-csd-reconstruction/paper_figures/deck_figures/pipeline_comparison/tiles/probability.png",286,224,112);
rect(286,224,112,112,null,[102,102,102],0.6);
text(342,340,[["P(transition line)",R]],10,[34,34,34],"c");
text(342,352,[["per pixel, 100 \u00D7 100",R]],10,[34,34,34],"c");
text(342,366,[["network output",B]],10,[8,81,156],"c");
var hl = rect(281,219,122,162,null,[8,81,156],1.3); hl.strokeDashes=[3.5,2.5];
arrow(402,280,418,280,[8,81,156],1.3,8);
box(422,224,118,112,[8,81,156],[234,242,250]);
text(481,260.2,[["threshold",B]],11.5,[34,34,34],"c");
text(481,273.4,[["chosen on",R]],10.5,[34,34,34],"c");
text(481,286.6,[["validation devices",R]],10.5,[34,34,34],"c");
text(481,340,[["scored on the lines",R]],10,[102,102,102],"c");
text(481,352,[["(F1@\u03C4)",R]],10,[102,102,102],"c");
arrow(544,280,560,280,[8,81,156],1.3,8);
place("F:/Seminar Fujita/projects/probabilistic-csd-reconstruction/paper_figures/deck_figures/pipeline_comparison/tiles/lines_pred.png",564,224,112);
rect(564,224,112,112,null,[102,102,102],0.6);
text(620,340,[["predicted lines",R]],10,[102,102,102],"c");

var base = "F:/Seminar Fujita/projects/probabilistic-csd-reconstruction/paper_figures/deck_figures/pipeline_comparison/pipeline_comparison_ref12";
var o = new IllustratorSaveOptions(); o.pdfCompatible = true;
doc.saveAs(new File(base + ".ai"), o);
var pdf = new PDFSaveOptions(); pdf.preserveEditability = false;
doc.saveAs(new File(base + ".pdf"), pdf);
var png = new ExportOptionsPNG24(); png.horizontalScale = 500; png.verticalScale = 500;
png.antiAliasing = true; png.transparency = false; png.artBoardClipping = true;
doc.exportFile(new File(base + ".png"), ExportType.PNG24, png);
var svg = new ExportOptionsSVG(); svg.embedRasterImages = true;
svg.fontType = SVGFontType.OUTLINEFONT;
doc.exportFile(new File(base + ".svg"), ExportType.SVG, svg);
doc.close(SaveOptions.DONOTSAVECHANGES);
"done";
