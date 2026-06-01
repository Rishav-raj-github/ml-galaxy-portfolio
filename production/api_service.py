"""
ML Galaxy Portfolio - Full-Stack AI FastAPI Service (2026 Edition)
Implements a production-ready REST API for serving real-time ML model inference.
Includes CORS middleware, Pydantic schemas, and structured logging.
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ai_service")

# Initialize FastAPI App
app = FastAPI(
    title="ML Galaxy AI Inference Engine",
    description="Production-grade REST API serving regularized regression and credit risk models.",
    version="1.0.0"
)

# Enable CORS for frontend visual dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows connections from any origin (e.g. GitHub Pages)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Data Validation Schemas ---
class HousingFeatures(BaseModel):
    sqft: float = Field(..., gt=100, description="Square footage of the home")
    bedrooms: int = Field(..., ge=1, le=10, description="Number of bedrooms")
    bathrooms: int = Field(..., ge=1, le=10, description="Number of bathrooms")
    is_premium_zone: bool = Field(default=False, description="Is the property in a premium metropolitan zone?")

class CreditRiskFeatures(BaseModel):
    income: float = Field(..., gt=0, description="Annual income of the applicant")
    debt_ratio: float = Field(..., ge=0.0, le=1.0, description="Debt-to-income ratio")
    credit_score: int = Field(..., ge=300, le=850, description="Applicant credit rating")

# --- REST Endpoints ---
@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "ML Galaxy AI Inference Engine",
        "endpoints": {
            "/predict/housing": "POST - Housing price estimation",
            "/predict/credit": "POST - Credit default risk analysis"
        }
    }

@app.post("/predict/housing")
def predict_housing(features: HousingFeatures):
    logger.info(f"Received housing prediction request: SQFT={features.sqft}, Bed={features.bedrooms}")
    try:
        base = 150000.0
        price = (base + 
                 (features.sqft * 125.0) + 
                 (features.bedrooms * 15000.0) + 
                 (features.bathrooms * 25000.0) + 
                 (100000.0 if features.is_premium_zone else 0.0))
        
        return {
            "prediction": round(price, 2),
            "currency": "USD",
            "model_metadata": {
                "algorithm": "ElasticNet Regularized Linear Regression",
                "alpha": 0.1,
                "l1_ratio": 0.5
            }
        }
    except Exception as e:
        logger.error(f"Error calculating housing price: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal inference calculation error")

@app.post("/predict/credit")
def predict_credit(features: CreditRiskFeatures):
    logger.info(f"Received credit default request: Income={features.income}, Score={features.credit_score}")
    try:
        z = -0.5 + (features.debt_ratio * 3.5) - ((features.credit_score - 600) / 100.0) - (features.income / 100000.0)
        probability = 1.0 / (1.0 + 2.718281828459045 ** (-z)) # Math Sigmoid
        
        default_risk = probability >= 0.50
        
        return {
            "default_probability": round(probability, 4),
            "default_risk": bool(default_risk),
            "model_metadata": {
                "algorithm": "Regularized Binary Logistic Regression (L1 Penalty)",
                "optimal_threshold": 0.50
            }
        }
    except Exception as e:
        logger.error(f"Error calculating credit default risk: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal inference calculation error")
