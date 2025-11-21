"""
FastAPI Application for African GDP Growth Prediction
Predicts GDP growth rates for African countries based on economic indicators
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import pickle
import numpy as np
from typing import List
import uvicorn
import os
import sys

# Print Python version for debugging
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")


# Define the custom LinearRegressionGD class (needed for unpickling)
class LinearRegressionGD:
    """
    Vectorized Linear Regression using Gradient Descent
    Optimized for speed using NumPy vectorization
    """
    
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.train_loss_history = []
        self.test_loss_history = []
    
    def fit(self, X_train, y_train, X_test=None, y_test=None):
        """Fit the model using gradient descent"""
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.iterations):
            # Predictions
            y_pred = X_train @ self.weights + self.bias
            
            # Compute gradients (vectorized)
            dw = (2/n_samples) * (X_train.T @ (y_pred - y_train))
            db = (2/n_samples) * np.sum(y_pred - y_train)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calculate losses
            train_loss = np.mean((y_pred - y_train) ** 2)
            self.train_loss_history.append(train_loss)
            
            if X_test is not None and y_test is not None:
                y_test_pred = X_test @ self.weights + self.bias
                test_loss = np.mean((y_test_pred - y_test) ** 2)
                self.test_loss_history.append(test_loss)
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

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
import os
import sys

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Custom unpickler to handle the LinearRegressionGD class
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'LinearRegressionGD':
            return LinearRegressionGD
        return super().find_class(module, name)

try:
    model_path = os.path.join(BASE_DIR, 'best_model.pkl')
    scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
    encoder_path = os.path.join(BASE_DIR, 'label_encoder.pkl')
    features_path = os.path.join(BASE_DIR, 'feature_names.pkl')
    metadata_path = os.path.join(BASE_DIR, 'model_metadata.pkl')
    
    print(f"Loading models from: {BASE_DIR}")
    
    with open(model_path, 'rb') as f:
        model = CustomUnpickler(f).load()
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Get list of valid countries
    valid_countries = list(label_encoder.classes_)
    
    print("✅ Model and preprocessors loaded successfully!")
    print(f"✅ Best Model: {metadata['best_model_name']}")
    print(f"✅ Test MSE: {metadata['test_mse']:.4f}")
    print(f"✅ Test R²: {metadata['test_r2']:.4f}")
    print(f"✅ Valid countries: {len(valid_countries)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print(f"❌ Current directory: {os.getcwd()}")
    print(f"❌ BASE_DIR: {BASE_DIR}")
    import traceback
    traceback.print_exc()
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
    
    @field_validator('country')
    @classmethod
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
