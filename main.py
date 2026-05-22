"""
Churn Prediction API

Run with:
    litestar --app main:app run --reload
Then open:
    http://localhost:8000/schema/swagger
"""

from litestar import Litestar, get, post
from pydantic import BaseModel

from app.logger_setup import setup_logging
from app.model_utils import predict_churn

logger = setup_logging()


# ---------------------------------------------------------------------------
# Request Schema
# ---------------------------------------------------------------------------
class ChurnRequest(BaseModel):
    credit_score: float
    age: float
    tenure: float
    balance: float
    num_of_products: float
    has_credit_card: float
    is_active_member: float
    estimated_salary: float
    geography_germany: float
    geography_spain: float
    gender_male: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@get("/")
def home() -> dict[str, str]:
    logger.info("Home endpoint accessed")
    return {"message": "Welcome to the churn prediction API"}


@get("/health")
def health() -> dict[str, str]:
    logger.info("Health endpoint accessed")
    return {"status": "healthy"}


@post("/predict")
def predict(data: ChurnRequest) -> dict[str, int]:
    features = list(data.model_dump().values())
    prediction = predict_churn(features)
    logger.info("Predict request received: %s", features)
    logger.info("Predict result: %s", prediction)
    return {"prediction": prediction}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Litestar(
    route_handlers=[home, health, predict],
)
