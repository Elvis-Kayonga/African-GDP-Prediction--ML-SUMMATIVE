"""
FastAPI Application for African GDP Growth Prediction
Predicts GDP growth rates for African countries based on economic indicators
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import pickle
import numpy as np
from typing import List
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="African GDP Growth Prediction API",
    description="Predicts GDP growth rates for African countries based on economic indicators",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and preprocessors at startup
try:
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    with open('model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    # Get list of valid countries
    valid_countries = list(label_encoder.classes_)
    
    print("✅ Model and preprocessors loaded successfully!")
    print(f"✅ Best Model: {metadata['best_model_name']}")
    print(f"✅ Test MSE: {metadata['test_mse']:.4f}")
    print(f"✅ Test R²: {metadata['test_r2']:.4f}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise


# Pydantic model for input validation
class EconomicIndicators(BaseModel):
    """
    Input model for GDP growth prediction with data type and range validation
    """
    year: int = Field(
        ...,
        ge=2000,
        le=2050,
        description="Year for prediction (2000-2050)"
    )
    inflation_rate: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Inflation rate in percentage (0-100%)"
    )
    unemployment_rate: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Unemployment rate in percentage (0-100%)"
    )
    fdi_millions_usd: float = Field(
        ...,
        ge=0.0,
        le=50000.0,
        description="Foreign Direct Investment in millions USD (0-50000)"
    )
    trade_balance_millions_usd: float = Field(
        ...,
        ge=-50000.0,
        le=50000.0,
        description="Trade balance in millions USD (-50000 to 50000)"
    )
    govt_debt_percent_gdp: float = Field(
        ...,
        ge=0.0,
        le=200.0,
        description="Government debt as percentage of GDP (0-200%)"
    )
    internet_penetration_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Internet penetration in percentage (0-100%)"
    )
    country: str = Field(
        ...,
        description="African country name (e.g., 'Nigeria', 'South Africa', 'Kenya')"
    )
    
    @validator('country')
    def validate_country(cls, v):
        """Validate that the country is in the valid list"""
        if v not in valid_countries:
            raise ValueError(
                f"Country '{v}' not found. Valid countries are: {', '.join(valid_countries)}"
            )
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "year": 2024,
                "inflation_rate": 12.5,
                "unemployment_rate": 18.0,
                "fdi_millions_usd": 3500.0,
                "trade_balance_millions_usd": -800.0,
                "govt_debt_percent_gdp": 38.0,
                "internet_penetration_percent": 55.0,
                "country": "Nigeria"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_gdp_growth_rate: float
    unit: str = "%"
    model_used: str
    input_data: dict
    
    class Config:
        schema_extra = {
            "example": {
                "predicted_gdp_growth_rate": 2.34,
                "unit": "%",
                "model_used": "Random Forest",
                "input_data": {
                    "year": 2024,
                    "country": "Nigeria",
                    "inflation_rate": 12.5
                }
            }
        }


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "African GDP Growth Prediction API",
        "version": "1.0.0",
        "description": "Predicts GDP growth rates for African countries",
        "endpoints": {
            "POST /predict": "Make a GDP growth prediction",
            "GET /countries": "Get list of supported countries",
            "GET /model-info": "Get information about the model",
            "GET /docs": "Swagger UI documentation",
            "GET /redoc": "ReDoc documentation"
        }
    }


@app.get("/countries")
async def get_countries():
    """Get list of supported African countries"""
    return {
        "countries": sorted(valid_countries),
        "total": len(valid_countries)
    }


@app.get("/model-info")
async def get_model_info():
    """Get information about the trained model"""
    return {
        "model_name": metadata['best_model_name'],
        "test_mse": round(metadata['test_mse'], 4),
        "test_r2_score": round(metadata['test_r2'], 4),
        "target_variable": metadata['target'],
        "features": metadata['features'],
        "supported_countries": len(valid_countries)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_gdp_growth(data: EconomicIndicators):
    """
    Predict GDP growth rate for an African country
    
    Returns:
        Predicted GDP growth rate in percentage
    """
    try:
        # Encode country
        country_encoded = label_encoder.transform([data.country])[0]
        
        # Create feature array in the correct order
        features = np.array([[
            data.year,
            data.inflation_rate,
            data.unemployment_rate,
            data.fdi_millions_usd,
            data.trade_balance_millions_usd,
            data.govt_debt_percent_gdp,
            data.internet_penetration_percent,
            country_encoded
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Return response
        return PredictionResponse(
            predicted_gdp_growth_rate=round(float(prediction), 2),
            unit="%",
            model_used=metadata['best_model_name'],
            input_data=data.dict()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "encoder_loaded": label_encoder is not None
    }


# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "prediction:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
