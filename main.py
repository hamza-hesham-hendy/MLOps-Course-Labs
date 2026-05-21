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
    standardscaler__CreditScore: float
    standardscaler__Age: float
    standardscaler__Tenure: float
    standardscaler__Balance: float
    standardscaler__NumOfProducts: float
    standardscaler__HasCrCard: float
    standardscaler__IsActiveMember: float
    standardscaler__EstimatedSalary: float
    onehotencoder__Geography_Germany: float
    onehotencoder__Geography_Spain: float
    onehotencoder__Gender_Male: float


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
    features = [
        data.standardscaler__CreditScore,
        data.standardscaler__Age,
        data.standardscaler__Tenure,
        data.standardscaler__Balance,
        data.standardscaler__NumOfProducts,
        data.standardscaler__HasCrCard,
        data.standardscaler__IsActiveMember,
        data.standardscaler__EstimatedSalary,
        data.onehotencoder__Geography_Germany,
        data.onehotencoder__Geography_Spain,
        data.onehotencoder__Gender_Male,
    ]
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
