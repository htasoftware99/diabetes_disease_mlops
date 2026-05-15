import os
import sys
import pickle
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from src.logger import get_logger
from src.custom_exception import CustomException

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Diabetes Prediction API")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
logger = get_logger(__name__)

# ── Model Load ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join("artifacts", "models", "diabetes.pkl")
SCALER_PATH = os.path.join("artifacts", "models", "diabetes_scaler.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    logger.info("Model and scaler loaded successfully.")
except Exception as e:
    logger.error(f"Model/scaler could not be loaded: {e}")
    model  = None
    scaler = None


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": None,
        "error":  None,
    })


@app.post("/", response_class=HTMLResponse)
async def predict_diabetes(
    request:                  Request,
    Pregnancies:              float = Form(...),
    Glucose:                  float = Form(...),
    BloodPressure:            float = Form(...),
    SkinThickness:            float = Form(...),
    Insulin:                  float = Form(...),
    BMI:                      float = Form(...),
    DiabetesPedigreeFunction: float = Form(...),
    Age:                      float = Form(...),
):
    result = None
    error  = None

    try:
        if model is None or scaler is None:
            raise CustomException("Model or scaler not available. Please check the logs for details.", sys)
        
        features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        features_scaled = scaler.transform(features)
        result = int(model.predict(features_scaled)[0])
        logger.info(f"Prediction made successfully: {result}")

    except CustomException as ce:
        error = str(ce)
        logger.error(f"CustomException in POST /: {error}")
    except Exception as e:
        error = f"An unexpected error occurred. Please check your entries: {str(e)}"
        logger.error(f"Unexpected error in POST /: {e}")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "error":  error,
    })


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)