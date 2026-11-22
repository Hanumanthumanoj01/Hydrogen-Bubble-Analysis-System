"""
AI-Based Hydrogen Bubble Analysis System — FastAPI Backend

HOW TO RUN (from the backend folder):  uvicorn main:app --port 8000
The React frontend (npm run dev in frontend/) talks to it on port 8000.

Frankfurt University of Applied Sciences · Fuel Cell Laboratory
Author: Manoj Hanumanthu  |  Supervisor: Prof. Dr. Enno Wagner
"""
""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
import cv2
import io
import base64
from pydantic import BaseModel
from typing import Optional
import uvicorn

from bubble_analysis import BubbleAnalyzer
import fisher_image
from constants import MM_PER_PIXEL
from simulator import ElectrolyzerSimulator

app = FastAPI(
    title="Hydrogen Bubble Analysis API",
    description="Entropy-information approach to bubble detection in alkaline water electrolysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = BubbleAnalyzer()
simulator = ElectrolyzerSimulator()


class SimulateRequest(BaseModel):
    current_density: float       # mA/cm²
    koh_concentration: float = 1.0   # mol/L
    temperature: float = 25.0        # °C
    electrode_type: str = "gde"      # gde | mesh | ans
    cathode_area: float = 2.54       # cm²


@app.get("/")
def root():
    # In the cloud container the built web application lives in static/;
    # there "/" opens the application. Locally "/" answers with the API status.
    import os
    from fastapi.responses import FileResponse
    index = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"status": "ok", "system": "Hydrogen Bubble Analysis API v1.0"}


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    current_density: float = Form(200.0),
    electrode_type: str = Form("gde"),
    cathode_area: float = Form(2.54),
    sensitivity: str = Form("medium"),
    detector: str = Form("standard"),
    mm_min: float = Form(0.20),
    mm_max: float = Form(0.90),
    min_pct: float = Form(3.0),
    max_pct: float = Form(60.0),
):
    """
    Analyze an uploaded microscope image.
    Returns Shannon entropy, bubble detections, size distribution,
    curvature, coverage, nucleation sites, H2 estimate, and overlay images.
    """
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = analyzer.full_analysis(
        img, current_density, electrode_type, cathode_area, sensitivity,
        detector=detector, mm_min=mm_min, mm_max=mm_max,
        min_pct=min_pct, max_pct=max_pct,
    )
    result["detector"] = detector
    return JSONResponse(content=result)


# ─────────────────────────────────────────────────────────────────────────────
#  RECORDINGS
# ─────────────────────────────────────────────────────────────────────────────
#
# Both video endpoints below call the same function, _analyse_recording. That
# is deliberate. The browser view and the downloadable zip must show the same
# detections for the same settings; if each endpoint carried its own copy of
# the loop they would drift apart the first time one of them was edited.

def _analyse_recording(path, current_density, electrode_type, cathode_area,
                       detector, min_pct, max_pct, every, max_frames,
                       camera_fps=0.0):
    """
    Decode a recording and analyse every `every`-th frame, up to `max_frames`.

    Returns (rows, overlays, fps, total, n_sites), where overlays maps a frame
    index to a full-resolution annotated PNG, base64-encoded, and fps is the
    frame rate used for the time column.

    FRAME RATE
    ----------
    The laboratory files store 24 fps in their container, while the camera
    recorded at the rate printed in the top-left corner of every frame
    ("640 x 480 / 200 fps / 20 us"). The burnt-in timestamps confirm the
    printed rate: on the 100 mA/cm2 recording five frames span 25 ms, i.e.
    200 fps. Times computed from the container rate are therefore too long by
    a factor of 200/24. When camera_fps is given it is used for every time
    value; otherwise the container rate is used and reported as such.

    HOW A RECORDING IS TREATED
    --------------------------
    Every sampled frame goes through the same detector the Image tab uses, so
    per-frame accuracy is identical to the still-image result by construction.

    For the "primary" detector the recording contributes one thing a still
    image cannot: the same bubble is seen in neighbouring frames. After every
    sampled frame has been detected, detect_primary.temporal_consensus()
    revises a frame whose circle for a site is more than 25 % off the size the
    neighbouring frames agree on (using a circle that frame itself fitted), and
    removes a fallback circle that no neighbouring frame confirms.

    An earlier version added bubbles at sites of high temporal activity that
    the still detector had missed. Since the still detector outlines whole
    bubbles, those additions were duplicates or plate, and they are no longer
    made; the number of active sites is still computed and returned.
    """
    import detect_video as dvid
    from detect_primary import detect_primary, DEFAULT_MM

    frames, fps, total = dvid.read_clip(path, max_frames=600)
    if camera_fps and camera_fps > 0:
        fps = float(camera_fps)
    every = max(1, int(every))

    sites, rmin, rmax = [], None, None
    if detector == "primary":
        H0, W0 = frames.shape[1:]
        rmin = max(0.5 * (min_pct / 100.0) * min(H0, W0), 6.0)
        rmax = 0.5 * (max_pct / 100.0) * min(H0, W0)
        act = dvid.activity_map(frames)
        strong = float(np.percentile(act, 97))
        sites = [s for s in dvid.find_sites(act, (H0, W0))
                 if act[s[1], s[0]] >= strong]

    # Pass 1: detection in every sampled frame. Pass 2 (primary detector only):
    # temporal_consensus() reconciles each frame with its neighbours, which
    # fixes single frames that chose an inner ring or a circle too large for
    # the bubble. Pass 3: metrics and overlays from the final circles.
    from detect_primary import temporal_consensus
    picked = list(range(0, len(frames), every))[:max(0, int(max_frames))]
    all_hits = []
    if detector == "primary":
        for idx in picked:
            gray = frames[idx]
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            hits = detect_primary(gray, DEFAULT_MM, MM_PER_PIXEL,
                                  size_frac=(min_pct / 100.0, max_pct / 100.0),
                                  bgr=bgr)
            # The active-site additions (dvid.detect_in_frame) are no longer
            # applied: since the rim stage outlines whole bubbles they added only
            # duplicates and plate - 75 extra circles over 60 frames of the
            # 688 mA/cm2 recording, whose 12 sites already carry 11 to 14
            # circles per frame. The site map is still computed and reported.
            all_hits.append(hits)
        temporal_consensus(all_hits)

    rows, overlays = [], {}
    for n, idx in enumerate(picked):
        gray = frames[idx]
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if detector == "primary":
            hits = sorted(all_hits[n], key=lambda c: -c["r_px"])
            bubbles = [(int(h["x"]), int(h["y"]), int(round(h["r_px"]))) for h in hits]
            scores = {b: float(min(1.0, h["agreement"]))
                      for b, h in zip(bubbles, hits)}
            Hval, Hn = analyzer.shannon_entropy(gray)
            metrics = analyzer.derive_metrics(bubbles, gray.shape,
                                              current_density, cathode_area)
            res = {
                "bubble_count": len(bubbles),
                "shannon_entropy": round(Hval, 4),
                "shannon_entropy_norm": round(Hn * 100, 2),
                "overlay_image": analyzer._make_overlay(bgr, bubbles, scores,
                                                        bold=True),
                **metrics,
            }
        else:
            res = analyzer.full_analysis(
                bgr, current_density, electrode_type, cathode_area,
                detector=detector, min_pct=min_pct, max_pct=max_pct,
            )

        rows.append({
            "frame": idx,
            "t_s": round(idx / fps, 3) if fps else None,
            "bubbles": res["bubble_count"],
            "H": res["shannon_entropy"],
            "H_norm": res["shannon_entropy_norm"],
            "mean_diameter_mm": res["mean_diameter_mm"],
            "coverage_pct": res["surface_coverage_pct"],
            "H2_mL_per_min": res["H2_production_mL_per_min"],
        })
        overlays[idx] = res["overlay_image"]

    return rows, overlays, fps, total, len(sites)


def _shrink_b64(b64, max_w=900, quality=82):
    """A viewing copy for the browser: width-capped JPEG. Sixty frames then cost
    a few megabytes rather than forty. The zip endpoint keeps full size."""
    raw = np.frombuffer(base64.b64decode(b64), np.uint8)
    im = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if im is None:
        return b64
    if im.shape[1] > max_w:
        sc = max_w / im.shape[1]
        im = cv2.resize(im, (max_w, int(im.shape[0] * sc)),
                        interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buf).decode() if ok else b64


async def _save_upload(file):
    import tempfile, os
    suffix = os.path.splitext(file.filename or "clip.avi")[1] or ".avi"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await file.read())
    tmp.close()
    return tmp.name


def _decode_error(name):
    return HTTPException(
        status_code=400,
        detail=f"Could not decode {name}. OpenCV opens AVI, MP4, MOV and MKV; "
               f"this file may be corrupt or use an unusual codec.")


@app.post("/analyze-video")
async def analyze_video(
    file: UploadFile = File(...),
    current_density: float = Form(200.0),
    electrode_type: str = Form("ans"),
    cathode_area: float = Form(2.54),
    detector: str = Form("primary"),
    min_pct: float = Form(3.0),
    max_pct: float = Form(60.0),
    every: int = Form(5),
    max_frames: int = Form(60),
    camera_fps: float = Form(0.0),
):
    """
    Run the detector over a recording, frame by frame, and return every sampled
    interval with its annotated frame.

    Decoding happens here with OpenCV rather than in the browser, because no
    browser decodes AVI; the laboratory's recordings therefore need no
    conversion, and MP4, MOV and MKV work the same way.
    """
    import os
    path = await _save_upload(file)
    try:
        try:
            rows, overlays, fps, total, n_sites = _analyse_recording(
                path, current_density, electrode_type, cathode_area,
                detector, min_pct, max_pct, every, max_frames, camera_fps)
        except IOError:
            raise _decode_error(file.filename)
        if not rows:
            raise HTTPException(status_code=400, detail="No frames could be read.")

        keys = sorted(overlays)
        intervals = [{**r, "image": _shrink_b64(overlays[r["frame"]])} for r in rows]
        picked = [keys[0], keys[len(keys) // 2], keys[-1]]
        previews = [{"frame": k, "image": overlays[k]} for k in dict.fromkeys(picked)]

        def mean(field):
            vals = [r[field] for r in rows if r[field] is not None]
            return round(float(np.mean(vals)), 4) if vals else None

        return JSONResponse(content={
            "filename": file.filename,
            "frames_in_file": total,
            "fps": round(fps, 2) if fps else None,
            "fps_source": "camera, entered" if camera_fps and camera_fps > 0
                          else "file container",
            "frames_analyzed": len(rows),
            "sampled_every": max(1, int(every)),
            "detector": detector,
            "sites": n_sites,
            "per_frame": rows,
            "intervals": intervals,
            "previews": previews,
            "mean_bubbles": mean("bubbles"),
            "mean_H": mean("H"),
            "mean_H_norm": mean("H_norm"),
            "mean_diameter_mm": mean("mean_diameter_mm"),
            "mean_coverage_pct": mean("coverage_pct"),
            "mean_H2_mL_per_min": mean("H2_mL_per_min"),
            "min_bubbles": min(r["bubbles"] for r in rows),
            "max_bubbles": max(r["bubbles"] for r in rows),
            "current_density": current_density,
            "cathode_area": cathode_area,
            "electrode_type": electrode_type,
        })
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@app.post("/analyze-video-frames")
async def video_frames_zip(
    file: UploadFile = File(...),
    current_density: float = Form(200.0),
    electrode_type: str = Form("ans"),
    cathode_area: float = Form(2.54),
    detector: str = Form("primary"),
    min_pct: float = Form(3.0),
    max_pct: float = Form(60.0),
    every: int = Form(5),
    max_frames: int = Form(60),
    camera_fps: float = Form(0.0),
):
    """
    The same analysis as /analyze-video, with the same settings, returned as a
    zip of full-resolution annotated PNGs plus per_frame.csv. Files are named
    by frame number and time so they sort correctly.

    The path starts with /analyze on purpose. The frontend's vite.config.js
    forwards /analyze, /simulate and /ml to the backend and nothing else, so a
    path outside those prefixes would reach the browser's dev server instead
    and the download would fail with no useful error.
    """
    import os, zipfile, csv, io as _io
    path = await _save_upload(file)
    try:
        try:
            rows, overlays, fps, total, n_sites = _analyse_recording(
                path, current_density, electrode_type, cathode_area,
                detector, min_pct, max_pct, every, max_frames, camera_fps)
        except IOError:
            raise _decode_error(file.filename)

        mem = _io.BytesIO()
        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
            for r in rows:
                t = "" if r["t_s"] is None else str(r["t_s"]).replace(".", "p")
                z.writestr(f"frame_{r['frame']:05d}_t{t}s.png",
                           base64.b64decode(overlays[r["frame"]]))
            buf = _io.StringIO()
            cols = ["frame", "t_s", "bubbles", "H", "H_norm",
                    "mean_diameter_mm", "coverage_pct", "H2_mL_per_min"]
            w = csv.writer(buf)
            w.writerow(cols)
            for r in rows:
                w.writerow([r[c] for c in cols])
            z.writestr("per_frame.csv", buf.getvalue())
            z.writestr("settings.txt",
                       f"file              {file.filename}\n"
                       f"detector          {detector}\n"
                       f"current density   {current_density} mA/cm2\n"
                       f"cathode area      {cathode_area} cm2\n"
                       f"electrode         {electrode_type}\n"
                       f"smallest bubble   {min_pct} % of frame height\n"
                       f"every             {every} frames\n"
                       f"max frames        {max_frames}\n"
                       f"frames in file    {total}\n"
                       f"fps               {fps}  ("
                       f"{'camera, entered' if camera_fps and camera_fps > 0 else 'file container'})\n")
        mem.seek(0)
        stem = os.path.splitext(os.path.basename(file.filename or "clip"))[0]
        return StreamingResponse(
            mem, media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{stem}_frames.zip"'})
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  HISTOGRAM AND FISHER INFORMATION FROM ONE PICTURE
# ─────────────────────────────────────────────────────────────────────────────

async def _read_image(file):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400,
                            detail=f"Could not read {file.filename} as an image.")
    return img


@app.post("/analyze-fisher")
async def analyze_fisher(
    file: UploadFile = File(...),
    n_peaks: int = Form(20),
    hist_smooth: float = Form(2.0),
    mask_timestamp: bool = Form(True),
):
    """
    Everything measurable from one still picture: the brightness histogram,
    the Shannon entropy and Fisher information of that histogram, the spatial
    Fisher maps F_x and F_y, and the strongest peaks with their significance.

    Nothing is detected. This is the detector-free view of the same picture,
    which is what makes it a check on the detector rather than a repeat of it.
    """
    img = await _read_image(file)
    out = fisher_image.analyse_image(img, n_peaks=n_peaks, hist_smooth=hist_smooth,
                                     mask_timestamp=mask_timestamp)
    out["filename"] = file.filename
    return JSONResponse(content=out)


@app.post("/analyze-fisher-xlsx")
async def analyze_fisher_xlsx(
    file: UploadFile = File(...),
    n_peaks: int = Form(20),
    hist_smooth: float = Form(2.0),
    mask_timestamp: bool = Form(True),
):
    """
    The same analysis as /analyze-fisher, as an Excel workbook with three
    sheets: Histogram (256 rows), Summary, and Peaks. Built here with openpyxl
    so the result does not depend on the browser's export helper.
    """
    import os, io as _io
    try:
        from openpyxl import Workbook
    except ImportError:
        raise HTTPException(status_code=500,
                            detail="Excel export needs openpyxl:  pip3 install openpyxl")
    img = await _read_image(file)
    r = fisher_image.analyse_image(img, n_peaks=n_peaks, hist_smooth=hist_smooth,
                                   mask_timestamp=mask_timestamp)

    wb = Workbook()
    ws = wb.active
    ws.title = "Histogram"
    ws.append(["brightness", "pixels"])
    for i, c in enumerate(r["histogram_counts"]):
        ws.append([i, c])

    ws = wb.create_sheet("Summary")
    ws.append(["quantity", "value"])
    for k, label in [
        ("width", "image width (px)"), ("height", "image height (px)"),
        ("masked_pixels", "pixels masked (burnt-in text)"),
        ("shannon_entropy", "Shannon entropy H (nats)"),
        ("shannon_entropy_norm", "H normalised (%)"),
        ("fisher_histogram_J", "Fisher information of histogram J"),
        ("histogram_smoothing", "histogram smoothing (bins)"),
        ("fisher_x_total", "Fisher total, x"), ("fisher_y_total", "Fisher total, y"),
        ("fisher_xy_ratio", "Fisher x / y ratio"),
        ("fisher_mean", "Fisher mean"), ("fisher_median", "Fisher median"),
        ("edge_flat_ratio", "edge / flat ratio"), ("edge_pixels", "Canny edge pixels"),
        ("brightness_floor", "brightness floor used"),
    ]:
        ws.append([label, r[k]])
    ws.append(["source file", file.filename])

    ws = wb.create_sheet("Peaks")
    ws.append(["rank", "x_px", "y_px", "fisher", "times_median", "brightness", "on_edge"])
    for i, p in enumerate(r["peaks"], 1):
        ws.append([i, p["x"], p["y"], p["F"], p["height_vs_median"], p["brightness"],
                   "yes" if p["on_edge"] else "no"])

    mem = _io.BytesIO()
    wb.save(mem)
    mem.seek(0)
    stem = os.path.splitext(os.path.basename(file.filename or "picture"))[0]
    return StreamingResponse(
        mem,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{stem}_fisher.xlsx"'})


@app.post("/simulate")
def simulate(req: SimulateRequest):
    """
    Return physically-grounded metrics for given electrolyzer parameters.
    Used by the Live Simulator tab.
    """
    result = simulator.compute(
        req.current_density, req.koh_concentration,
        req.temperature, req.electrode_type, req.cathode_area
    )
    return JSONResponse(content=result)


@app.get("/health")
def health():
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# Cloud deployment: serve the built React app from the same address.
#
# Locally the React dev server (npm run dev, port 5173) forwards the API calls
# to this backend. In the cloud container (Dockerfile in the project folder)
# the frontend is built once into backend/static/, and this backend serves it
# at "/" next to the API, so the whole application has ONE link and the
# frontend's relative calls ("/analyze", ...) reach this server unchanged.
# The mount is added last, so every API route above takes precedence.
# ---------------------------------------------------------------------------
import os as _os                                                   # noqa: E402
_STATIC = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "static")
if _os.path.isdir(_STATIC):
    from fastapi.staticfiles import StaticFiles                    # noqa: E402
    app.mount("/", StaticFiles(directory=_STATIC, html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(_os.environ.get("PORT", 8000)), reload=True)