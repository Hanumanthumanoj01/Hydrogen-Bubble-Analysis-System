"""
git remote add origin https://github.com/Hanumanthumanoj01/Hydrogen-Bubble-Analysis-System.git — FastAPI Backend
Frankfurt University of Applied Sciences · Fuel Cell Laboratory
Author: Manoj Hanumanthu  |  Supervisor: Prof. Dr. Enno Wagner
"""
""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import io
import base64
from pydantic import BaseModel
from typing import Optional
import uvicorn

import os
import tempfile

from bubble_analysis import BubbleAnalyzer
from simulator import ElectrolyzerSimulator
from video_analysis import analyze_video

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
    return {"status": "ok", "system": "Hydrogen Bubble Analysis API v1.0"}


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    current_density: float = Form(200.0),
    electrode_type: str = Form("gde"),
    cathode_area: float = Form(2.54),
    sensitivity: str = Form("medium")
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
        img, current_density, electrode_type, cathode_area, sensitivity
    )
    return JSONResponse(content=result)


@app.post("/analyze_video")
async def analyze_video_endpoint(
    file: UploadFile = File(...),
    current_density: float = Form(900.0),
    cathode_area: float = Form(2.54),
    thresh_percentile: float = Form(88.0),
    frame_step: int = Form(1),
    min_persistence: int = Form(2),
    use_rolling_bg: bool = Form(False),
):
    """
    Analyze an uploaded electrolysis VIDEO using temporal background
    subtraction (see video_analysis.py for why this is the primary method
    for video rather than the single-frame detector used by /analyze).

    The static electrode is reconstructed as the per-pixel temporal median
    and subtracted, which removes scratches / mesh / specular hot-spots that
    are indistinguishable from bubble rims in any single frame.

    Returns per-frame metrics, a summary including bubble growth-rate and
    lifetime statistics from tracking, and the computed background image.
    """
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty video upload")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        result = analyze_video(
            tmp_path,
            current_density=current_density,
            cathode_area=cathode_area,
            thresh_percentile=thresh_percentile,
            frame_step=frame_step,
            min_persistence=min_persistence,
            use_rolling_bg=use_rolling_bg,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return JSONResponse(content=result)


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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)