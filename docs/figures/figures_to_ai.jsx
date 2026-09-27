/*  figures_to_ai.jsx — turn the LaTeX figures into Illustrator .ai files.
 *
 *  Run it from Illustrator:  File > Scripts > Other Script...  (Ctrl+F12)
 *
 *  It reads the *_paths.svg beside it -- dvisvgm's conversion of the LaTeX
 *  DVI, in which every glyph is already a path -- and writes a real .ai next
 *  to each.  Nothing depends on Computer Modern being installed on the
 *  machine that opens the file: by the time Illustrator sees it, the maths
 *  is geometry.
 *
 *  USE THIS, NOT THE PDF, IF YOU WANT TO MOVE THE LABELS.  In the SVG each
 *  label is one named group -- c_s1g1, c_d1g3, Vg1, matrices -- so it drags
 *  as one object and the Layers panel says which it is.  A PDF cannot do
 *  that: TeX sets "c_{s_1g_1}" in three fonts (cmmi10, cmmi7, cmr5), which
 *  Illustrator imports as three separate text objects.  The .ai beside each
 *  figure is a copy of the PDF, so it has the same split; it is there for a
 *  quick look and for keeping the labels as live type.
 *
 *  Rebuild the inputs after editing a .tex:
 *      pdflatex dqd_model_sketch.tex
 *      latex    dqd_model_sketch.tex
 *      dvisvgm  --no-fonts --exact --output=dqd_model_sketch_paths.svg \
 *               dqd_model_sketch.dvi
 *      then this script.
 */
#target illustrator

var HERE = new File($.fileName).parent.fsName + "/";

var FIGURES = [
    "dqd_model_sketch",        // the sensor above, the dots below
    "dqd_model_sketch_color",  // the same, colour-coded by what a coupling does
    "dqd_model_full",          // the wide-arc arrangement
    "dqd_model_full_color"     // the same, colour-coded
];

function toAi(name) {
    var src = new File(HERE + name + "_paths.svg");
    if (!src.exists) return name + ": no " + name + "_paths.svg";

    var doc = app.open(src);

    // Everything on one layer, grouped, so the figure moves as one object
    // when it is dropped into a manuscript layout.
    doc.selection = null;
    var items = doc.pageItems;
    if (items.length > 1) {
        for (var i = 0; i < items.length; i++) items[i].selected = true;
        var g = doc.groupItems.add();
        var sel = doc.selection;
        for (var j = sel.length - 1; j >= 0; j--) sel[j].moveToBeginning(g);
        doc.selection = null;
    }

    var opts = new IllustratorSaveOptions();
    opts.compatibility = Compatibility.ILLUSTRATOR17;   // CC 2013+, as the
    opts.pdfCompatible = true;                          // other figures use
    doc.saveAs(new File(HERE + name + ".ai"), opts);

    var w = Math.round(doc.width), h = Math.round(doc.height);
    doc.close(SaveOptions.DONOTSAVECHANGES);
    return name + ".ai  (" + w + " x " + h + " pt)";
}

var done = [];
for (var k = 0; k < FIGURES.length; k++) done.push(toAi(FIGURES[k]));
done.join("\n");
