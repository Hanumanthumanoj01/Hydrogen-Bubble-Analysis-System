"""
build_report.py
===============
Builds report.pdf and report.docx from report.md.

    python3 build_report.py

Both files get the same Frankfurt UAS front page, which Prof. Wagner asked
for in his comments of 22 September 2026:

    "Make an entire front page with University Logo, Study course,
     supervisor, date, etc."

WHY THIS SCRIPT EXISTS
----------------------
A front page cannot be written once and used for both formats. PDF comes
out of LaTeX, where the page is laid out with \\includegraphics and
centring commands; Word has no idea what those mean and pandoc silently
drops them. So the two are built separately:

    PDF   pandoc inserts titlepage.tex ahead of the body as raw LaTeX.
    DOCX  pandoc builds the body, then python-docx puts a native Word
          title page in front of it, with the logo as a real picture.

The body text, the figures and the numbers are identical in both. Only the
first page is produced twice.

REQUIREMENTS
------------
    pandoc            (with a LaTeX engine: xelatex, or pdflatex)
    pip3 install python-docx

If python-docx is missing the DOCX is still written, just with pandoc's
plain title block instead of the designed front page. The script says so
rather than failing.
"""

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__)) or "."
MD = os.path.join(HERE, "report.md")
TEX = os.path.join(HERE, "titlepage.tex")
LOGO = os.path.join(HERE, "figures", "fra-uas-logo.png")

PDF = os.path.join(HERE, "report.pdf")
DOCX = os.path.join(HERE, "report.docx")

TITLE = "AI-Based Hydrogen Bubble Analysis System"
SUBTITLE = ("An Entropy-Information Approach to Bubble Detection\n"
            "in Alkaline Water Electrolysis")

FRONT_ROWS = [
    ("Submitted by", "Manoj Hanumanthu"),
    ("Study course", "M.Sc. Information Technology, 4th semester"),
    ("Supervisor",   "Prof. Dr. Enno Wagner"),
    ("Laboratory",   "Fuel Cell Laboratory"),
    ("University",   "Frankfurt University of Applied Sciences"),
    ("Date",         "24 September 2026"),
]


def run(cmd):
    print("  $ " + " ".join(os.path.basename(c) if i == 0 else c
                            for i, c in enumerate(cmd)))
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
    return r.returncode == 0


def pick_engine():
    for e in ("xelatex", "pdflatex", "lualatex"):
        if shutil.which(e):
            return e
    return None


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def have_sty(name):
    """Is this LaTeX package installed? kpsewhich is how TeX itself looks."""
    if not shutil.which("kpsewhich"):
        return True          # cannot tell; assume a complete installation
    r = subprocess.run(["kpsewhich", name], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() != ""


def first_font(*names):
    """The first of these fonts the system actually has, or None."""
    if not shutil.which("fc-list"):
        return None
    r = subprocess.run(["fc-list", ":", "family"], capture_output=True,
                       text=True)
    have = r.stdout
    for n in names:
        if n in have:
            return n
    return None


def build_pdf():
    engine = pick_engine()
    if engine is None:
        print("  no LaTeX engine found (xelatex / pdflatex) - skipping PDF")
        print("  install one, e.g.  brew install --cask mactex-no-gui")
        return False

    cmd = ["pandoc", "report.md",
           "--include-before-body", "titlepage.tex",
           "--pdf-engine", engine,
           "-V", "colorlinks=true",
           "-V", "linkcolor=blue",
           # Every caption in report.md already begins "Figure N - ...",
           # which is what Word shows. LaTeX would add its own "Figure N:"
           # in front of that and print the number twice, so its automatic
           # label is switched off for figures only. Tables keep theirs.
           "-V", r"header-includes=\usepackage{caption}",
           "-V", r"header-includes=\captionsetup[figure]{labelformat=empty}"]

    # Pandoc's default template loads lmodern unless a font family is named.
    # A full TeX Live or MacTeX has lmodern and this branch never runs; some
    # trimmed-down Linux installations do not, and without this the build
    # stops on "File `lmodern.sty' not found" with no clue why.
    if not have_sty("lmodern.sty"):
        print("  lmodern.sty missing - substituting an available font")
        cmd += ["-V", "fontfamily=mathptmx"]
        font = first_font("DejaVu Serif", "Liberation Serif", "FreeSerif")
        if font:
            cmd += ["-V", "mainfont=" + font]

    cmd += ["-o", "report.pdf"]
    return run(cmd)


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

def build_docx():
    if not run(["pandoc", "report.md", "-o", "report.docx"]):
        return False

    try:
        import docx
        from docx.shared import Pt, Cm, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
    except ImportError:
        print("  python-docx not installed, so the DOCX keeps pandoc's plain")
        print("  title block. To get the designed front page:")
        print("      pip3 install python-docx")
        return True

    d = docx.Document(DOCX)
    body = d.element.body

    def para(text="", size=11, bold=False, align="center", space_after=6,
             colour=None):
        p = d.add_paragraph()
        p.alignment = {"center": WD_ALIGN_PARAGRAPH.CENTER,
                       "left": WD_ALIGN_PARAGRAPH.LEFT}[align]
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.space_before = Pt(0)
        for i, line in enumerate(text.split("\n")):
            r = p.add_run(line)
            r.font.size = Pt(size)
            r.bold = bold
            if colour:
                r.font.color.rgb = RGBColor(*colour)
            if i < len(text.split("\n")) - 1:
                r.add_break()
        return p

    # Everything below is appended to the end of the document first, then
    # moved to the front in one block. Building it in place would mean
    # fighting python-docx's insert API for every paragraph.
    made = []

    if os.path.exists(LOGO):
        p = d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(26)
        p.add_run().add_picture(LOGO, width=Cm(10.5))
        made.append(p)

    made.append(para("Faculty 2 — Computer Science and Engineering", 13,
                     space_after=2))
    made.append(para("Fuel Cell Laboratory", 13, space_after=30))

    made.append(para("_" * 62, 11, colour=(0x88, 0x88, 0x88), space_after=14))
    made.append(para(TITLE, 21, bold=True, space_after=12))
    made.append(para(SUBTITLE, 13, space_after=14))
    made.append(para("_" * 62, 11, colour=(0x88, 0x88, 0x88), space_after=34))

    made.append(para("Individual Project", 14, bold=True, space_after=6))
    made.append(para("submitted in partial fulfilment of the requirements "
                     "for the degree of", 11, space_after=2))
    made.append(para("Master of Science in Information Technology", 11,
                     bold=True, space_after=34))

    t = d.add_table(rows=0, cols=2)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for label, value in FRONT_ROWS:
        row = t.add_row().cells
        rl = row[0].paragraphs[0].add_run(label)
        rl.bold = True
        rl.font.size = Pt(11)
        row[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
        rv = row[1].paragraphs[0].add_run("   " + value)
        rv.font.size = Pt(11)
    made.append(t)

    made.append(para("", 11, space_after=30))
    made.append(para("Building on the method proposed in\n"
                     "Wagner, Tennert, Probsthain & Mishra, "
                     "Heat and Mass Transfer 61, 108 (2025)", 9,
                     space_after=0))

    brk = d.add_paragraph()
    brk.add_run().add_break(WD_BREAK.PAGE)
    made.append(brk)

    # Move the block to the very top, keeping its internal order.
    first = body[0]
    for item in made:
        el = item._element if hasattr(item, "_element") else item._tbl
        body.remove(el)
        first.addprevious(el)

    d.save(DOCX)
    print("  front page inserted into report.docx")
    return True


def main():
    if not shutil.which("pandoc"):
        print("pandoc is not installed. Get it from https://pandoc.org/")
        sys.exit(1)
    if not os.path.exists(MD):
        print("report.md not found next to this script.")
        sys.exit(1)
    if not os.path.exists(LOGO):
        print("WARNING: figures/fra-uas-logo.png is missing.")
        print("The front page will be built without the university logo.")

    print("BUILDING THE REPORT")
    print("=" * 62)

    print("\nPDF:")
    ok_pdf = build_pdf()

    print("\nDOCX:")
    ok_docx = build_docx()

    print("\n" + "=" * 62)
    for ok, path in ((ok_pdf, PDF), (ok_docx, DOCX)):
        if ok and os.path.exists(path):
            kb = os.path.getsize(path) / 1024
            print(f"  wrote {os.path.basename(path):<12} {kb:>8.0f} KB")
        else:
            print(f"  FAILED {os.path.basename(path)}")


if __name__ == "__main__":
    main()
