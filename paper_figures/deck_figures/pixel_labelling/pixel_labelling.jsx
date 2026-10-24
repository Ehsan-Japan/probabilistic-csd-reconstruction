
var doc = app.documents.add(DocumentColorSpace.RGB, 590.0, 350.0);
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

rect(58,264,34,34,[230,237,245],[181,181,181],0.5);
rect(58,230,34,34,[230,237,245],[181,181,181],0.5);
rect(58,196,34,34,[230,237,245],[181,181,181],0.5);
rect(58,162,34,34,[230,237,245],[181,181,181],0.5);
rect(58,128,34,34,[230,237,245],[181,181,181],0.5);
rect(58,94,34,34,[158,202,225],[8,81,156],1.1);
rect(58,60,34,34,[251,234,214],[181,181,181],0.5);
rect(58,26,34,34,[251,234,214],[181,181,181],0.5);
rect(92,264,34,34,[230,237,245],[181,181,181],0.5);
rect(92,230,34,34,[230,237,245],[181,181,181],0.5);
rect(92,196,34,34,[230,237,245],[181,181,181],0.5);
rect(92,162,34,34,[230,237,245],[181,181,181],0.5);
rect(92,128,34,34,[158,202,225],[8,81,156],1.1);
rect(92,94,34,34,[251,234,214],[181,181,181],0.5);
rect(92,60,34,34,[251,234,214],[181,181,181],0.5);
rect(92,26,34,34,[251,234,214],[181,181,181],0.5);
rect(126,264,34,34,[230,237,245],[181,181,181],0.5);
rect(126,230,34,34,[230,237,245],[181,181,181],0.5);
rect(126,196,34,34,[230,237,245],[181,181,181],0.5);
rect(126,162,34,34,[158,202,225],[8,81,156],1.1);
rect(126,128,34,34,[251,234,214],[181,181,181],0.5);
rect(126,94,34,34,[251,234,214],[181,181,181],0.5);
rect(126,60,34,34,[251,234,214],[181,181,181],0.5);
rect(126,26,34,34,[251,234,214],[181,181,181],0.5);
rect(160,264,34,34,[230,237,245],[181,181,181],0.5);
rect(160,230,34,34,[230,237,245],[181,181,181],0.5);
rect(160,196,34,34,[230,237,245],[181,181,181],0.5);
rect(160,162,34,34,[158,202,225],[8,81,156],1.1);
rect(160,128,34,34,[251,234,214],[181,181,181],0.5);
rect(160,94,34,34,[251,234,214],[181,181,181],0.5);
rect(160,60,34,34,[251,234,214],[181,181,181],0.5);
rect(160,26,34,34,[251,234,214],[181,181,181],0.5);
rect(194,264,34,34,[230,237,245],[181,181,181],0.5);
rect(194,230,34,34,[230,237,245],[181,181,181],0.5);
rect(194,196,34,34,[158,202,225],[8,81,156],1.1);
rect(194,162,34,34,[251,234,214],[181,181,181],0.5);
rect(194,128,34,34,[251,234,214],[181,181,181],0.5);
rect(194,94,34,34,[251,234,214],[181,181,181],0.5);
rect(194,60,34,34,[251,234,214],[181,181,181],0.5);
rect(194,26,34,34,[251,234,214],[181,181,181],0.5);
rect(228,264,34,34,[230,237,245],[181,181,181],0.5);
rect(228,230,34,34,[230,237,245],[181,181,181],0.5);
rect(228,196,34,34,[158,202,225],[8,81,156],1.1);
rect(228,162,34,34,[251,234,214],[181,181,181],0.5);
rect(228,128,34,34,[251,234,214],[181,181,181],0.5);
rect(228,94,34,34,[251,234,214],[181,181,181],0.5);
rect(228,60,34,34,[251,234,214],[181,181,181],0.5);
rect(228,26,34,34,[251,234,214],[181,181,181],0.5);
rect(262,264,34,34,[230,237,245],[181,181,181],0.5);
rect(262,230,34,34,[158,202,225],[8,81,156],1.1);
rect(262,196,34,34,[251,234,214],[181,181,181],0.5);
rect(262,162,34,34,[251,234,214],[181,181,181],0.5);
rect(262,128,34,34,[251,234,214],[181,181,181],0.5);
rect(262,94,34,34,[251,234,214],[181,181,181],0.5);
rect(262,60,34,34,[251,234,214],[181,181,181],0.5);
rect(262,26,34,34,[251,234,214],[181,181,181],0.5);
rect(296,264,34,34,[230,237,245],[181,181,181],0.5);
rect(296,230,34,34,[158,202,225],[8,81,156],1.1);
rect(296,196,34,34,[251,234,214],[181,181,181],0.5);
rect(296,162,34,34,[251,234,214],[181,181,181],0.5);
rect(296,128,34,34,[251,234,214],[181,181,181],0.5);
rect(296,94,34,34,[251,234,214],[181,181,181],0.5);
rect(296,60,34,34,[251,234,214],[181,181,181],0.5);
rect(296,26,34,34,[251,234,214],[181,181,181],0.5);
rect(58,94,34,34,null,[8,81,156],1.1);
rect(92,128,34,34,null,[8,81,156],1.1);
rect(126,162,34,34,null,[8,81,156],1.1);
rect(160,162,34,34,null,[8,81,156],1.1);
rect(194,196,34,34,null,[8,81,156],1.1);
rect(228,196,34,34,null,[8,81,156],1.1);
rect(262,230,34,34,null,[8,81,156],1.1);
rect(296,230,34,34,null,[8,81,156],1.1);
dot(75,281,1.6,[34,34,34]);
dot(75,247,1.6,[34,34,34]);
dot(75,213,1.6,[34,34,34]);
dot(75,179,1.6,[34,34,34]);
dot(75,145,1.6,[34,34,34]);
dot(75,111,1.6,[34,34,34]);
dot(75,77,1.6,[34,34,34]);
dot(75,43,1.6,[34,34,34]);
dot(109,281,1.6,[34,34,34]);
dot(109,247,1.6,[34,34,34]);
dot(109,213,1.6,[34,34,34]);
dot(109,179,1.6,[34,34,34]);
dot(109,145,1.6,[34,34,34]);
dot(109,111,1.6,[34,34,34]);
dot(109,77,1.6,[34,34,34]);
dot(109,43,1.6,[34,34,34]);
dot(143,281,1.6,[34,34,34]);
dot(143,247,1.6,[34,34,34]);
dot(143,213,1.6,[34,34,34]);
dot(143,179,1.6,[34,34,34]);
dot(143,145,1.6,[34,34,34]);
dot(143,111,1.6,[34,34,34]);
dot(143,77,1.6,[34,34,34]);
dot(143,43,1.6,[34,34,34]);
dot(177,281,1.6,[34,34,34]);
dot(177,247,1.6,[34,34,34]);
dot(177,213,1.6,[34,34,34]);
dot(177,179,1.6,[34,34,34]);
dot(177,145,1.6,[34,34,34]);
dot(177,111,1.6,[34,34,34]);
dot(177,77,1.6,[34,34,34]);
dot(177,43,1.6,[34,34,34]);
dot(211,281,1.6,[34,34,34]);
dot(211,247,1.6,[34,34,34]);
dot(211,213,1.6,[34,34,34]);
dot(211,179,1.6,[34,34,34]);
dot(211,145,1.6,[34,34,34]);
dot(211,111,1.6,[34,34,34]);
dot(211,77,1.6,[34,34,34]);
dot(211,43,1.6,[34,34,34]);
dot(245,281,1.6,[34,34,34]);
dot(245,247,1.6,[34,34,34]);
dot(245,213,1.6,[34,34,34]);
dot(245,179,1.6,[34,34,34]);
dot(245,145,1.6,[34,34,34]);
dot(245,111,1.6,[34,34,34]);
dot(245,77,1.6,[34,34,34]);
dot(245,43,1.6,[34,34,34]);
dot(279,281,1.6,[34,34,34]);
dot(279,247,1.6,[34,34,34]);
dot(279,213,1.6,[34,34,34]);
dot(279,179,1.6,[34,34,34]);
dot(279,145,1.6,[34,34,34]);
dot(279,111,1.6,[34,34,34]);
dot(279,77,1.6,[34,34,34]);
dot(279,43,1.6,[34,34,34]);
dot(313,281,1.6,[34,34,34]);
dot(313,247,1.6,[34,34,34]);
dot(313,213,1.6,[34,34,34]);
dot(313,179,1.6,[34,34,34]);
dot(313,145,1.6,[34,34,34]);
dot(313,111,1.6,[34,34,34]);
dot(313,77,1.6,[34,34,34]);
dot(313,43,1.6,[34,34,34]);
line([[58,100.8],[330,250.4]],[192,57,43],1.8);
boxed(text(324,31,[["(",R],["N",I],["1",I,1],[" + 1, ",R],["N",I],["2",I,1],[")",R]],9,[34,34,34],"r"));
boxed(text(63,282,[["(",R],["N",I],["1",I,1],[", ",R],["N",I],["2",I,1],[")",R]],9,[34,34,34],"l"));
text(167,184,[["P",B]],9,[8,81,156],"c");
arrow(58,312,344,312,[34,34,34],0.8,6);
text(348,305,[["V",I],["1",I,1]],10,[34,34,34],"l");
arrow(44,298,44,14,[34,34,34],0.8,6);
text(44,0,[["V",I],["2",I,1]],10,[34,34,34],"c");
line([[58,324],[92,324]],[102,102,102],0.8);
line([[58,321],[58,327]],[102,102,102],0.8);
line([[92,321],[92,327]],[102,102,102],0.8);
text(98,319,[["1 pixel = unit of distance (incl. \u03C4)",R]],8,[102,102,102],"l");
rect(388,30,14,14,[230,237,245],[181,181,181],0.5);
rect(402,30,14,14,[251,234,214],[181,181,181],0.5);
text(424,31,[["pixel colour: charge configuration",R]],9,[34,34,34],"l");
rect(395,54,14,14,[230,237,245],[181,181,181],0.5);
dot(402,61,1.6,[34,34,34]);
text(424,50,[["pixel centre: the single voltage",R]],9,[34,34,34],"l");
text(424,61,[["pair the simulator evaluates",R]],9,[34,34,34],"l");
line([[388,87],[416,87]],[192,57,43],1.8);
text(424,81,[["true transition line (zero width)",R]],9,[34,34,34],"l");
rect(395,104,14,14,[158,202,225],[8,81,156],1.1);
text(424,100,[["labelled pixel: last one before the",R]],9,[34,34,34],"l");
text(424,111,[["configuration changes",R]],9,[34,34,34],"l");
rect(422,162,30,30,[251,234,214],[181,181,181],0.5);
rect(452,192,30,30,[251,234,214],[181,181,181],0.5);
rect(452,162,30,30,[251,234,214],[181,181,181],0.5);
rect(422,192,30,30,[158,202,225],[8,81,156],1.1);
text(429,210,[["P",B]],9,[8,81,156],"c");
dot(437,207,1.6,[34,34,34]);
dot(467,207,1.6,[34,34,34]);
dot(437,177,1.6,[34,34,34]);
arrow(440,207,464,207,[34,34,34],0.8,5);
arrow(437,204,437,180,[34,34,34],0.8,5);
text(487,201,[["+",R],["V",I],["1",I,1]],8,[34,34,34],"l");
text(437,149,[["+",R],["V",I],["2",I,1]],8,[34,34,34],"c");
line([[418,181.8],[484,228]],[192,57,43],1.8);
text(388,234,[["P is labelled if its configuration differs",R]],8.5,[34,34,34],"l");
text(388,245,[["from its +",R],["V",I],["1",I,1],[" or its +",R],["V",I],["2",I,1],[" neighbour",R]],8.5,[34,34,34],"l");

var base = "F:/Seminar Fujita/projects/probabilistic-csd-reconstruction/paper_figures/deck_figures/pixel_labelling/pixel_labelling";
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
