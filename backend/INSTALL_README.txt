SCRIPTS BUNDLE - where these files go
=====================================

Unzip this straight into your backend folder:

    ~/Downloads/hydrogen-bubble-analysis/backend/

All the .py files sit next to main.py.  Overwrite when asked - these are
the current versions.

    backend/
      main.py                  (already there)
      simulator.py             (already there)
      ml_model.py              (already there)
      ml_routes.py             (already there)
      train_from_videos.py     (already there - yours works, keep it)

      constants.py             <- from this bundle
      bubble_analysis.py       <- from this bundle
      video_analysis.py        <- from this bundle
      fisher_analysis.py       <- from this bundle
      fisher_pixel.py          <- NEW, Prof. Wagner's method
      run_all.py               <- from this bundle
      make_flow_diagram.py     <- from this bundle
      make_report_figure.py    <- from this bundle
      make_session_figure.py   <- NEW, Figure 13 with error bars
      make_filter_figures.py   <- from this bundle
      make_placeholders.py     <- from this bundle
      build_report.py          <- NEW, builds report.pdf and report.docx
      titlepage.tex            <- NEW, the front page for the PDF


WHAT IS NEW IN THIS VERSION
---------------------------

fisher_pixel.py
    The method Prof. Wagner proposed in his comments of 22 September:
    Fisher information straight from the pixel brightness, with no
    detector involved at all.  F = (change in brightness)^2 / brightness.
    Makes Figures 15 to 18.

make_session_figure.py
    Figure 13, rebuilt the way he asked for it: error bars, no lines
    between the points, and larger lettering on the axes.  This figure
    previously had no script at all, so it could not be regenerated.

build_report.py + titlepage.tex
    Builds report.pdf and report.docx from report.md, both with the
    Frankfurt UAS front page.  Put report.md, titlepage.tex and the
    figures/ folder next to it.


ONE-TIME INSTALL
----------------

    pip3 install matplotlib scipy pandas scikit-learn joblib python-docx

For the PDF you also need pandoc and a LaTeX engine:

    brew install pandoc
    brew install --cask mactex-no-gui        (large, ~4 GB)

If you only want the Word file, pandoc alone is enough - build_report.py
says so and carries on rather than failing.


CHECK IT WORKED
---------------

    cd ~/Downloads/hydrogen-bubble-analysis/backend
    ls fisher_pixel.py
    python3 make_flow_diagram.py

That should print three lines and create a report_figures/ folder.


THE FULL SEQUENCE
-----------------

    cd ~/Downloads/hydrogen-bubble-analysis/backend

    # diagrams - Figures 1, 2, 4
    python3 make_flow_diagram.py

    # background and filter figures - Figures 7, 8, 9, 10
    python3 make_filter_figures.py session_C/bubble_video_2283mV_688mA_41deg.avi

    # per-file panels and summary.csv - one session at a time
    python3 run_all.py session_A    &&  mv results results_A
    python3 run_all.py session_B    &&  mv results results_B
    python3 run_all.py session_C    &&  mv results results_C
    python3 run_all.py screenshots  &&  mv results results_screenshots

    # main results - Figure 12
    python3 make_report_figure.py

    # session comparison with error bars - Figure 13
    python3 make_session_figure.py

    # bubble-table Fisher maps - Figure 11
    python3 fisher_analysis.py session_C --filter-tiny

    # pixel-level Fisher information - Figures 15 to 18
    python3 fisher_pixel.py session_B
    python3 fisher_pixel.py session_C

    # model numbers for Section 5.7
    python3 train_from_videos.py videos

    # rebuild the report itself
    python3 build_report.py

Order matters in one place only: make_session_figure.py reads the
summary.csv files, so the four run_all.py lines have to come first.

Everything lands in  report_figures/,  results_A/,  results_B/,
results_C/,  fisher_output/  and  fisher_pixel_output_<folder>/.


WHAT THE NEW FISHER SCRIPT PRINTS
---------------------------------

    j (mA/cm2)  frames   mean F_t  edge/flat  masked px
           100      90  3.643e-03       1.32        841
           300     100  2.943e-04       2.08       1224
           600      90  1.110e-02       1.00        837
           900      90  6.116e-03       1.52       1116

"edge/flat" is the number that matters.  It is how much more Fisher
information sits on the brightness edges - the bubble rims - than on the
flat areas.  Above about 1.5 means the method is finding the ring-shaped
contact area Prof. Wagner described.  1.00 means it is finding nothing.

"masked px" is the camera's own timestamp.  Its millisecond digits change
in every single frame, so without excluding them they would be the
brightest thing in every map.  The script finds them automatically and
reports how many pixels it removed.

If your numbers differ from the table above, that is fine as long as the
recordings differ.  If you run the same four videos and get different
numbers, something is wrong - tell me.


IF A COMMAND FAILS
------------------

"can't open file ... No such file or directory"
    You are in the wrong folder, or the script is not there.
    Run:   pwd            (must end in /backend)
           ls *.py        (must list the script)

"ModuleNotFoundError: No module named 'matplotlib'"
    Run the pip3 install line above.

"ImportError: cannot import name 'classify_anchored_bubbles'"
    video_analysis.py was not replaced. Copy it again from this bundle.

"No videos found"
    The folder you named does not exist, or the filenames have no
    "NNNmA" in them. Check with:   ls session_C/

"results_A/summary.csv is required and was not found"
    make_session_figure.py was run before run_all.py. Run the four
    run_all.py lines first.

"File `lmodern.sty' not found"
    Your LaTeX installation is incomplete. build_report.py works around
    this automatically; if it still fails, the Word file is built either
    way and you can export a PDF from Word.
