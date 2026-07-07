"""
ml_routes.py — ML API routes, mounted onto main FastAPI app.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from pydantic import BaseModel
from typing import Optional

from bubble_analysis1 import BubbleAnalyzer
from ml_model import BubbleMLModel

router = APIRouter(prefix="/ml", tags=["Machine Learning"])
analyzer = BubbleAnalyzer()
ml = BubbleMLModel()


class LabelRequest(BaseModel):
    current_density: float
    electrode_type: str   # gde | mesh | ans


@router.post("/add-sample")
async def add_sample(
    file: UploadFile = File(...),
    current_density: float = 200.0,
    electrode_type: str = "gde",
    cathode_area: float = 2.54,
):
    """Upload a labelled image to grow the training dataset."""
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    analysis = analyzer.full_analysis(img, current_density, electrode_type, cathode_area)
    result = ml.add_sample(analysis, current_density, electrode_type)
    return JSONResponse(content=result)


@router.post("/train")
def train(use_xgboost: bool = False):
    """Train the bubble classifier on the current labelled dataset."""
    return JSONResponse(content=ml.train(use_xgboost=use_xgboost))


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_density: float = 200.0,
    electrode_type: str = "gde",
    cathode_area: float = 2.54,
):
    """Predict operating regime from a bubble image."""
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    analysis = analyzer.full_analysis(img, current_density, electrode_type, cathode_area)
    prediction = ml.predict(analysis)
    return JSONResponse(content={**analysis, **prediction})


@router.get("/feature-importance")
def feature_importance():
    """Return trained model feature importances."""
    return JSONResponse(content=ml.feature_importance())


@router.get("/dataset-info")
def dataset_info():
    return JSONResponse(content={
        "sample_count": len(ml.dataset),
        "model_ready": ml.model is not None,
    })
