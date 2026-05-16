# import os
# import sys
# import pickle
# import numpy as np
# import uvicorn
# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
# from src.logger import get_logger
# from src.custom_exception import CustomException

# # ── App Setup ──────────────────────────────────────────────────────────────────
# app = FastAPI(title="Diabetes Prediction API")
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")
# logger = get_logger(__name__)

# # ── Model Load ─────────────────────────────────────────────────────────────────
# MODEL_PATH = os.path.join("artifacts", "models", "diabetes.pkl")
# SCALER_PATH = os.path.join("artifacts", "models", "diabetes_scaler.pkl")

# try:
#     with open(MODEL_PATH, "rb") as f:
#         model = pickle.load(f)
#     with open(SCALER_PATH, "rb") as f:
#         scaler = pickle.load(f)
#     logger.info("Model and scaler loaded successfully.")
# except Exception as e:
#     logger.error(f"Model/scaler could not be loaded: {e}")
#     model  = None
#     scaler = None


# # ── Routes ─────────────────────────────────────────────────────────────────────
# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     return templates.TemplateResponse("index.html", {
#         "request": request,
#         "result": None,
#         "error":  None,
#     })


# @app.post("/", response_class=HTMLResponse)
# async def predict_diabetes(
#     request:                  Request,
#     Pregnancies:              float = Form(...),
#     Glucose:                  float = Form(...),
#     BloodPressure:            float = Form(...),
#     SkinThickness:            float = Form(...),
#     Insulin:                  float = Form(...),
#     BMI:                      float = Form(...),
#     DiabetesPedigreeFunction: float = Form(...),
#     Age:                      float = Form(...),
# ):
#     result = None
#     error  = None

#     try:
#         if model is None or scaler is None:
#             raise CustomException("Model or scaler not available. Please check the logs for details.", sys)
        
#         features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
#         features_scaled = scaler.transform(features)
#         result = int(model.predict(features_scaled)[0])
#         logger.info(f"Prediction made successfully: {result}")

#     except CustomException as ce:
#         error = str(ce)
#         logger.error(f"CustomException in POST /: {error}")
#     except Exception as e:
#         error = f"An unexpected error occurred. Please check your entries: {str(e)}"
#         logger.error(f"Unexpected error in POST /: {e}")

#     return templates.TemplateResponse("index.html", {
#         "request": request,
#         "result": result,
#         "error":  error,
#     })


# # ── Entry Point ────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)




# import os
# import sys
# import pickle
# import numpy as np
# import pandas as pd
# import uvicorn
# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
# from alibi_detect.cd import KSDrift
# from src.logger import get_logger
# from src.custom_exception import CustomException
# from src.feature_store import RedisFeatureStore
# from sklearn.preprocessing import StandardScaler
# from prometheus_client import start_http_server, Counter, Gauge

# # ── App Setup ──────────────────────────────────────────────────────────────────
# app = FastAPI(title="Diabetes Prediction API")
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")
# logger = get_logger(__name__)

# # ── Constants ──────────────────────────────────────────────────────────────────
# MODEL_PATH  = os.path.join("artifacts", "models", "diabetes.pkl")
# SCALER_PATH = os.path.join("artifacts", "models", "diabetes_scaler.pkl")

# FEATURE_NAMES = [
#     "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
#     "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
# ]

# # ── Model & Scaler Load ────────────────────────────────────────────────────────
# try:
#     with open(MODEL_PATH, "rb") as f:
#         model = pickle.load(f)
#     with open(SCALER_PATH, "rb") as f:
#         scaler = pickle.load(f)
#     logger.info("Model and scaler loaded successfully.")
# except Exception as e:
#     logger.error(f"Model/scaler could not be loaded: {e}")
#     model  = None
#     scaler = None

# # ── KSDrift Setup ──────────────────────────────────────────────────────────────
# ksd = None

# def setup_drift_detector():
#     """
#     Reads the training features stored in Redis during the Data Processing
#     pipeline, scales them with the already-fitted scaler, and initialises
#     the KSDrift detector using those values as the reference distribution.
#     """
#     global ksd
#     try:
#         feature_store = RedisFeatureStore()
#         entity_ids    = feature_store.get_all_entity_ids()

#         if not entity_ids:
#             logger.warning("No entities found in Redis. Drift detector will not be initialised.")
#             return

#         all_features = feature_store.get_batch_features(entity_ids)
#         ref_df = pd.DataFrame.from_dict(all_features, orient="index")[FEATURE_NAMES]

#         # Use the same scaler that was fitted during data processing
#         if scaler is not None:
#             ref_scaled = scaler.transform(ref_df)
#         else:
#             logger.warning("Scaler is not available; using raw features as reference.")
#             ref_scaled = ref_df.values

#         ksd = KSDrift(x_ref=ref_scaled, p_val=0.05)
#         logger.info(f"KSDrift detector initialised with {len(entity_ids)} reference samples.")

#     except Exception as e:
#         logger.error(f"Failed to initialise KSDrift detector: {e}")
#         ksd = None

# # Initialise drift detector at startup
# setup_drift_detector()


# def detect_drift(features_scaled: np.ndarray) -> bool:
#     """
#     Runs the KSDrift test on the scaled incoming features.
#     Returns True if drift is detected, False otherwise.
#     """
#     if ksd is None:
#         logger.warning("KSDrift detector is not available; skipping drift check.")
#         return False

#     drift_response = ksd.predict(features_scaled)
#     drift_data     = drift_response.get("data", {})
#     is_drift       = drift_data.get("is_drift", 0)

#     if is_drift == 1:
#         logger.warning("Data drift detected in the incoming request!")
#         # Per-feature p-values (useful for debugging which features drifted)
#         p_vals = drift_data.get("p_val", [])
#         logger.warning(f"Per-feature p-values: {dict(zip(FEATURE_NAMES, p_vals))}")
#     else:
#         logger.info("No data drift detected.")

#     return bool(is_drift)


# # ── Routes ─────────────────────────────────────────────────────────────────────
# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     return templates.TemplateResponse("index.html", {
#         "request":    request,
#         "result":     None,
#         "error":      None,
#         "drift_warning": False,
#     })


# @app.post("/", response_class=HTMLResponse)
# async def predict_diabetes(
#     request:                  Request,
#     Pregnancies:              float = Form(...),
#     Glucose:                  float = Form(...),
#     BloodPressure:            float = Form(...),
#     SkinThickness:            float = Form(...),
#     Insulin:                  float = Form(...),
#     BMI:                      float = Form(...),
#     DiabetesPedigreeFunction: float = Form(...),
#     Age:                      float = Form(...),
# ):
#     result        = None
#     error         = None
#     drift_warning = False

#     try:
#         if model is None or scaler is None:
#             raise CustomException(
#                 "Model or scaler not available. Please check the logs for details.", sys
#             )

#         # Build feature array
#         features = np.array([[
#             Pregnancies, Glucose, BloodPressure, SkinThickness,
#             Insulin, BMI, DiabetesPedigreeFunction, Age
#         ]])
#         features_scaled = scaler.transform(features)

#         # ── Drift Detection ────────────────────────────────────────────────────
#         drift_warning = detect_drift(features_scaled)

#         # ── Prediction ─────────────────────────────────────────────────────────
#         result = int(model.predict(features_scaled)[0])
#         logger.info(f"Prediction: {result} | Drift: {drift_warning}")

#     except CustomException as ce:
#         error = str(ce)
#         logger.error(f"CustomException in POST /: {error}")
#     except Exception as e:
#         error = f"An unexpected error occurred. Please check your entries: {str(e)}"
#         logger.error(f"Unexpected error in POST /: {e}")

#     return templates.TemplateResponse("index.html", {
#         "request":       request,
#         "result":        result,
#         "error":         error,
#         "drift_warning": drift_warning,
#     })


# # ── Entry Point ────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)








import os
import sys
import pickle
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from alibi_detect.cd import KSDrift
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from src.logger import get_logger
from src.custom_exception import CustomException
from src.feature_store import RedisFeatureStore
from sklearn.preprocessing import StandardScaler

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Diabetes Prediction API")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
logger = get_logger(__name__)

# ── Prometheus Metrics ─────────────────────────────────────────────────────────
def _get_or_create(metric_class, name, description, **kwargs):
    try:
        return metric_class(name, description, **kwargs)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name + "_total") \
            or REGISTRY._names_to_collectors.get(name)

prediction_count          = _get_or_create(Counter,    "prediction_count",          "Total number of predictions made")
drift_count               = _get_or_create(Counter,    "drift_count",               "Total number of times data drift was detected")
positive_prediction_count = _get_or_create(Counter,    "positive_prediction_count", "Total number of positive (diabetes) predictions")
negative_prediction_count = _get_or_create(Counter,    "negative_prediction_count", "Total number of negative (no diabetes) predictions")
prediction_latency        = _get_or_create(Histogram,  "prediction_latency_seconds","Time spent processing a prediction request",
                                           buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5])

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join("artifacts", "models", "diabetes.pkl")
SCALER_PATH = os.path.join("artifacts", "models", "diabetes_scaler.pkl")

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# ── Model & Scaler Load ────────────────────────────────────────────────────────
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

# ── KSDrift Setup ──────────────────────────────────────────────────────────────
ksd = None

def setup_drift_detector():
    """
    Reads the training features stored in Redis during the Data Processing
    pipeline, scales them with the already-fitted scaler, and initialises
    the KSDrift detector using those values as the reference distribution.
    """
    global ksd
    try:
        feature_store = RedisFeatureStore()
        entity_ids    = feature_store.get_all_entity_ids()

        if not entity_ids:
            logger.warning("No entities found in Redis. Drift detector will not be initialised.")
            return

        all_features = feature_store.get_batch_features(entity_ids)
        ref_df = pd.DataFrame.from_dict(all_features, orient="index")[FEATURE_NAMES]

        # Use the same scaler that was fitted during data processing
        if scaler is not None:
            ref_scaled = scaler.transform(ref_df)
        else:
            logger.warning("Scaler is not available; using raw features as reference.")
            ref_scaled = ref_df.values

        ksd = KSDrift(x_ref=ref_scaled, p_val=0.05)
        logger.info(f"KSDrift detector initialised with {len(entity_ids)} reference samples.")

    except Exception as e:
        logger.error(f"Failed to initialise KSDrift detector: {e}")
        ksd = None

# Initialise drift detector at startup
setup_drift_detector()


def detect_drift(features_scaled: np.ndarray) -> bool:
    """
    Runs the KSDrift test on the scaled incoming features.
    Returns True if drift is detected, False otherwise.
    """
    if ksd is None:
        logger.warning("KSDrift detector is not available; skipping drift check.")
        return False

    drift_response = ksd.predict(features_scaled)
    drift_data     = drift_response.get("data", {})
    is_drift       = drift_data.get("is_drift", 0)

    if is_drift == 1:
        logger.warning("Data drift detected in the incoming request!")
        # Per-feature p-values (useful for debugging which features drifted)
        p_vals = drift_data.get("p_val", [])
        logger.warning(f"Per-feature p-values: {dict(zip(FEATURE_NAMES, p_vals))}")
    else:
        logger.info("No data drift detected.")

    return bool(is_drift)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request":       request,
        "result":        None,
        "error":         None,
        "drift_warning": False,
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
    result        = None
    error         = None
    drift_warning = False

    try:
        if model is None or scaler is None:
            raise CustomException(
                "Model or scaler not available. Please check the logs for details.", sys
            )

        # Build feature array
        features = np.array([[
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age
        ]])
        features_scaled = scaler.transform(features)

        # ── Drift Detection ────────────────────────────────────────────────────
        with prediction_latency.time():

            # ── Drift Detection ────────────────────────────────────────────────
            drift_warning = detect_drift(features_scaled)
            if drift_warning:
                drift_count.inc()

            # ── Prediction ────────────────────────────────────────────────────
            result = int(model.predict(features_scaled)[0])

        prediction_count.inc()
        if result == 1:
            positive_prediction_count.inc()
        else:
            negative_prediction_count.inc()

        logger.info(f"Prediction: {result} | Drift: {drift_warning}")

    except CustomException as ce:
        error = str(ce)
        logger.error(f"CustomException in POST /: {error}")
    except Exception as e:
        error = f"An unexpected error occurred. Please check your entries: {str(e)}"
        logger.error(f"Unexpected error in POST /: {e}")

    return templates.TemplateResponse("index.html", {
        "request":       request,
        "result":        result,
        "error":         error,
        "drift_warning": drift_warning,
    })


# ── Prometheus Metrics Endpoint ────────────────────────────────────────────────
@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)