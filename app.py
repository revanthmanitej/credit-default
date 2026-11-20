"""
Credit Card Default Risk API (FastAPI)

Serves the model saved by credit_default_risk_optimized.ipynb
(best_credit_default_model.joblib: Random Forest, original features, cost-optimal threshold).

Run locally:
    pip install -r requirements.txt
    uvicorn app:app --reload
    -> interactive docs at http://127.0.0.1:8000/docs

Environment variables (all optional):
    MODEL_PATH     path to the .joblib bundle   (default: best_credit_default_model.joblib next to app.py)
    ALLOW_ORIGINS  comma-separated CORS origins (default: *)

This is a human-review screening aid, NOT an automated lending decision.
Only load .joblib files you created or trust: joblib/pickle can execute code when loading.
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

logger = logging.getLogger("credit-risk-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).resolve().parent / "best_credit_default_model.joblib"))
STATE: Dict[str, Any] = {"bundle": None}

DISCLAIMER = (
    "Screening aid only. A flag means 'send for human risk review', not an automatic credit rejection. "
    "Cost figures are illustrative project assumptions, not real bank losses."
)

# Same bands as the notebook (descriptive only; the threshold defines the decision)
RISK_BANDS = [
    {"band": "Low", "probability_range": "< 0.10"},
    {"band": "Moderate", "probability_range": "0.10 to < 0.25"},
    {"band": "High", "probability_range": "0.25 to < 0.50"},
    {"band": "Very high", "probability_range": ">= 0.50"},
]


# ---------------------------------------------------------------------------
# Feature engineering: identical to make_features() in the notebook
# ---------------------------------------------------------------------------
PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
AMOUNT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]


def make_features(data: pd.DataFrame, approach: str) -> pd.DataFrame:
    result = data.copy()

    if approach == "original":
        return result

    if approach == "average":
        groups = [(PAY_COLS, "avg_pay_status"), (BILL_COLS, "avg_bill_amount"), (AMOUNT_COLS, "avg_payment_amount")]
        for cols, new_col in groups:
            available = [c for c in cols if c in result.columns]
            result[new_col] = result[available].mean(axis=1)
            result = result.drop(columns=available)
        return result

    if approach == "feature_engineered":
        pay = [c for c in PAY_COLS if c in result]
        bills = [c for c in BILL_COLS if c in result]
        payments = [c for c in AMOUNT_COLS if c in result]
        if pay:
            result["avg_pay_status"] = result[pay].mean(axis=1)
            result["worst_pay_status"] = result[pay].max(axis=1)
            result["months_late"] = (result[pay] > 0).sum(axis=1)
            result["recent_pay_status"] = result[pay[0]]
        if bills:
            result["avg_bill_amount"] = result[bills].mean(axis=1)
            result["bill_amount_change"] = result[bills[0]] - result[bills[-1]]
        if payments:
            result["avg_payment_amount"] = result[payments].mean(axis=1)
            result["total_payment_amount"] = result[payments].sum(axis=1)
        if bills and payments:
            result["overall_payment_bill_ratio"] = result["avg_payment_amount"] / (result["avg_bill_amount"].abs() + 1)
        if "LIMIT_BAL" in result and bills:
            result["average_credit_utilization"] = result["avg_bill_amount"] / (result["LIMIT_BAL"].abs() + 1)
        return result

    raise ValueError(f"Unknown feature approach: {approach}")


# ---------------------------------------------------------------------------
# Model loading and scoring helpers
# ---------------------------------------------------------------------------
def load_bundle() -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}. Put best_credit_default_model.joblib next to app.py "
                           f"or set MODEL_PATH.")
    bundle = joblib.load(MODEL_PATH)
    required = {"model", "model_name", "feature_approach", "threshold", "input_columns", "feature_columns"}
    missing = required - set(bundle)
    if missing:
        raise RuntimeError(f"Model bundle is missing keys: {sorted(missing)}")
    STATE["bundle"] = bundle
    logger.info("Loaded model '%s' (approach=%s, threshold=%.2f) from %s",
                bundle["model_name"], bundle["feature_approach"], bundle["threshold"], MODEL_PATH)
    return bundle


def get_bundle() -> Dict[str, Any]:
    bundle = STATE.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return bundle


def risk_band(probability: float) -> str:
    if probability < 0.10:
        return "Low"
    if probability < 0.25:
        return "Moderate"
    if probability < 0.50:
        return "High"
    return "Very high"


def prepare_features(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Raw customer dicts -> model-ready feature frame (same cleaning as training)."""
    bundle = get_bundle()
    df = pd.DataFrame(records)[bundle["input_columns"]].astype(float)
    # Training-time cleaning of undocumented category codes
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    features = make_features(df, bundle["feature_approach"])
    return features.reindex(columns=bundle["feature_columns"])


def default_probabilities(features: pd.DataFrame) -> np.ndarray:
    model = get_bundle()["model"]
    positive_index = list(model.classes_).index(1)
    return model.predict_proba(features)[:, positive_index]


def build_result(probability: float, threshold: float) -> Dict[str, Any]:
    flagged = probability >= threshold
    return {
        "default_probability": round(float(probability), 4),
        "risk_band": risk_band(float(probability)),
        "threshold_used": round(float(threshold), 4),
        "flagged": bool(flagged),
        "decision": "FLAG FOR HUMAN RISK REVIEW" if flagged else "BELOW SELECTED RISK THRESHOLD",
    }


def resolve_threshold(override: Optional[float]) -> float:
    return float(override) if override is not None else float(get_bundle()["threshold"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
EXAMPLE_CUSTOMER = {
    "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
    "PAY_1": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
    "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
    "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0,
}


class Customer(BaseModel):
    """One customer's original input columns (the same 23 columns used for training)."""
    model_config = ConfigDict(populate_by_name=True, json_schema_extra={"example": EXAMPLE_CUSTOMER})

    LIMIT_BAL: float = Field(..., gt=0, description="Credit limit (NT$)")
    SEX: int = Field(..., ge=1, le=2, description="1 = male, 2 = female")
    EDUCATION: int = Field(..., ge=0, le=6, description="1 = graduate school, 2 = university, 3 = high school, 4/others = other")
    MARRIAGE: int = Field(..., ge=0, le=3, description="1 = married, 2 = single, 3/0 = other")
    AGE: int = Field(..., ge=18, le=120, description="Age in years")

    PAY_1: int = Field(..., ge=-2, le=9, validation_alias=AliasChoices("PAY_1", "PAY_0"),
                       description="Most recent repayment status: -1 paid on time, 1..9 months late (-2/0 = no balance / minimum paid). Alias: PAY_0")
    PAY_2: int = Field(..., ge=-2, le=9, description="Repayment status, 2 months ago")
    PAY_3: int = Field(..., ge=-2, le=9, description="Repayment status, 3 months ago")
    PAY_4: int = Field(..., ge=-2, le=9, description="Repayment status, 4 months ago")
    PAY_5: int = Field(..., ge=-2, le=9, description="Repayment status, 5 months ago")
    PAY_6: int = Field(..., ge=-2, le=9, description="Repayment status, 6 months ago")

    BILL_AMT1: float = Field(..., description="Bill statement, most recent month (can be negative for credits)")
    BILL_AMT2: float = Field(..., description="Bill statement, 2 months ago")
    BILL_AMT3: float = Field(..., description="Bill statement, 3 months ago")
    BILL_AMT4: float = Field(..., description="Bill statement, 4 months ago")
    BILL_AMT5: float = Field(..., description="Bill statement, 5 months ago")
    BILL_AMT6: float = Field(..., description="Bill statement, 6 months ago")

    PAY_AMT1: float = Field(..., ge=0, description="Amount paid, most recent month")
    PAY_AMT2: float = Field(..., ge=0, description="Amount paid, 2 months ago")
    PAY_AMT3: float = Field(..., ge=0, description="Amount paid, 3 months ago")
    PAY_AMT4: float = Field(..., ge=0, description="Amount paid, 4 months ago")
    PAY_AMT5: float = Field(..., ge=0, description="Amount paid, 5 months ago")
    PAY_AMT6: float = Field(..., ge=0, description="Amount paid, 6 months ago")


class PredictionResponse(BaseModel):
    default_probability: float
    risk_band: str
    threshold_used: float
    flagged: bool
    decision: str
    note: str = DISCLAIMER


class BatchRequest(BaseModel):
    customers: List[Customer] = Field(..., min_length=1, max_length=1000)


class BatchItem(BaseModel):
    index: int
    default_probability: float
    risk_band: str
    threshold_used: float
    flagged: bool
    decision: str


class BatchResponse(BaseModel):
    total: int
    flagged: int
    flag_rate: float
    results: List[BatchItem]
    note: str = DISCLAIMER


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    load_bundle()  # fail fast at startup if the model file is missing or malformed
    yield


app = FastAPI(
    title="Credit Card Default Risk API",
    version="1.0.0",
    description=(
        "Scores a customer's probability of defaulting next month and flags customers for human review "
        "using a cost-optimised decision threshold. " + DISCLAIMER
    ),
    lifespan=lifespan,
)

_origins = [o.strip() for o in os.getenv("ALLOW_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["info"])
def root() -> Dict[str, Any]:
    return {"service": "Credit Card Default Risk API", "docs": "/docs", "health": "/health",
            "model_info": "/model-info", "predict": "POST /predict", "batch": "POST /predict/batch"}


@app.get("/health", tags=["info"])
def health() -> Dict[str, Any]:
    loaded = STATE.get("bundle") is not None
    return {"status": "ok" if loaded else "model_not_loaded", "model_loaded": loaded}


@app.get("/model-info", tags=["info"])
def model_info() -> Dict[str, Any]:
    bundle = get_bundle()
    threshold = float(bundle["threshold"])
    costs = bundle.get("cost_assumptions", {})
    return {
        "model_name": bundle["model_name"],
        "feature_approach": bundle["feature_approach"],
        "default_threshold": threshold,
        "cost_assumptions": costs,
        "input_columns": bundle["input_columns"],
        "n_model_features": len(bundle["feature_columns"]),
        "risk_bands": RISK_BANDS,
        "decision_rule": (f"Flag for human review if predicted default probability >= {threshold:.2f}. "
                          f"Threshold chosen on validation data to minimise cost "
                          f"(false alarm = {costs.get('FP', '?')}, missed default = {costs.get('FN', '?')})."),
        "disclaimer": DISCLAIMER,
    }


@app.get("/feature-importance", tags=["info"])
def feature_importance(top: int = Query(10, ge=1, le=50, description="How many features to return")) -> Dict[str, Any]:
    """Global (model-wide) feature importance, for explainability. Not a per-customer explanation."""
    bundle = get_bundle()
    model = bundle["model"]
    final = model.steps[-1][1] if hasattr(model, "steps") else model
    importance = getattr(final, "feature_importances_", None)
    if importance is None and hasattr(final, "coef_"):
        importance = np.abs(np.ravel(final.coef_))
    names = bundle["feature_columns"]
    if importance is None or len(importance) != len(names):
        raise HTTPException(status_code=404, detail="Feature importance is not available for this model type.")
    ranked = sorted(zip(names, importance), key=lambda kv: kv[1], reverse=True)[:top]
    return {"model_name": bundle["model_name"],
            "features": [{"feature": n, "importance": round(float(v), 4)} for n, v in ranked]}


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict(
    customer: Customer,
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0,
                                       description="Optional override of the decision threshold. Defaults to the cost-optimal threshold saved with the model."),
) -> Dict[str, Any]:
    """Score one customer."""
    t = resolve_threshold(threshold)
    features = prepare_features([customer.model_dump()])
    probability = float(default_probabilities(features)[0])
    logger.info("Scored 1 customer (band=%s)", risk_band(probability))
    return build_result(probability, t)


@app.post("/predict/batch", response_model=BatchResponse, tags=["scoring"])
def predict_batch(
    request: BatchRequest,
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0,
                                       description="Optional override of the decision threshold."),
) -> Dict[str, Any]:
    """Score up to 1,000 customers in one call."""
    t = resolve_threshold(threshold)
    features = prepare_features([c.model_dump() for c in request.customers])
    probabilities = default_probabilities(features)
    results = [{"index": i, **build_result(float(p), t)} for i, p in enumerate(probabilities)]
    flagged = sum(r["flagged"] for r in results)
    logger.info("Scored %d customers (%d flagged)", len(results), flagged)
    return {"total": len(results), "flagged": flagged, "flag_rate": round(flagged / len(results), 4), "results": results}
