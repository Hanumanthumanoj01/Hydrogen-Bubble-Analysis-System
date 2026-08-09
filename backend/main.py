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

from bubble_analysis import BubbleAnalyzer
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