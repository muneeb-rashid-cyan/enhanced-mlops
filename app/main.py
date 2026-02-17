"""
Enhanced FastAPI ML API with multiple endpoints
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import time
from datetime import datetime
from typing import Dict, List
import logging
from contextlib import asynccontextmanager

from app.schemas import (
    IrisInput, WineInput, BatchPredictionInput,
    PredictionOutput, BatchPredictionOutput, ModelInfo,
    HealthResponse, MetricsResponse, DatasetType, ModelType
)
from app.model_service import ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model_manager: ModelManager = None
app_start_time = time.time()
prediction_count = 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global model_manager
    
    # Startup
    logger.info("🚀 Starting Enhanced MLOps API...")
    model_manager = ModelManager()
    logger.info(f"Loaded {len(model_manager.get_available_models())} models")
    yield
    
    # Shutdown
    logger.info("Shutting down Enhanced MLOps API...")

# Create FastAPI app
app = FastAPI(
    title="Enhanced MLOps API",
    description="Professional ML API with multiple models and endpoints",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get model manager
def get_model_manager() -> ModelManager:
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    return model_manager

@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint"""
    return {
        "message": "Enhanced MLOps API is running!",
        "version": "1.0.0",
        "models_available": len(model_manager.get_available_models()) if model_manager else 0,
        "endpoints": [
            "/health", "/metrics", "/models", 
            "/predict/iris", "/predict/wine", "/predict/batch",
            "/docs"
        ]
    }

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check(manager: ModelManager = Depends(get_model_manager)):
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=len(manager.get_available_models()),
        version="1.0.0"
    )

@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics(manager: ModelManager = Depends(get_model_manager)):
    """Get API metrics"""
    uptime = time.time() - app_start_time
    stats = manager.get_stats()
    total_predictions = sum(stats.values())
    
    return MetricsResponse(
        total_predictions=total_predictions,
        models_served=stats,
        average_response_time=0.1,  # Simplified
        uptime_seconds=uptime
    )

@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models(manager: ModelManager = Depends(get_model_manager)):
    """List all available models"""
    models = []
    for model_key in manager.get_available_models():
        model_info = manager.get_model_info(model_key)
        if model_info:
            models.append(model_info)
    return models

@app.post("/predict/iris", response_model=PredictionOutput, tags=["Prediction"])
async def predict_iris(
    input_data: IrisInput,
    model_type: ModelType = ModelType.logistic,
    manager: ModelManager = Depends(get_model_manager)
):
    """Predict iris flower type"""
    global prediction_count
    
    try:
        # Prepare features
        features = np.array([[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]])
        
        # Make prediction
        model_key = f"iris_{model_type.value}"
        prediction, confidence, probabilities = manager.predict(model_key, features)
        
        prediction_count += 1
        
        return PredictionOutput(
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Iris prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict/wine", response_model=PredictionOutput, tags=["Prediction"])
async def predict_wine(
    input_data: WineInput,
    model_type: ModelType = ModelType.logistic,
    manager: ModelManager = Depends(get_model_manager)
):
    """Predict wine class"""
    global prediction_count
    
    try:
        # Prepare features
        features = np.array([[
            input_data.alcohol, input_data.malic_acid, input_data.ash,
            input_data.alcalinity_of_ash, input_data.magnesium, input_data.total_phenols,
            input_data.flavanoids, input_data.nonflavanoid_phenols, input_data.proanthocyanins,
            input_data.color_intensity, input_data.hue, input_data.od280_od315_of_diluted_wines,
            input_data.proline
        ]])
        
        # Make prediction
        model_key = f"wine_{model_type.value}"
        prediction, confidence, probabilities = manager.predict(model_key, features)
        
        prediction_count += 1
        
        return PredictionOutput(
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Wine prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/predict/examples", tags=["Examples"])
async def get_examples():
    """Get example inputs for testing"""
    return {
        "iris_examples": [
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},  # Setosa
            {"sepal_length": 6.2, "sepal_width": 2.9, "petal_length": 4.3, "petal_width": 1.3},  # Versicolor
            {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}   # Virginica
        ],
        "wine_examples": [
            {
                "alcohol": 13.20, "malic_acid": 1.78, "ash": 2.14, "alcalinity_of_ash": 11.2,
                "magnesium": 100.0, "total_phenols": 2.65, "flavanoids": 2.76, "nonflavanoid_phenols": 0.26,
                "proanthocyanins": 1.28, "color_intensity": 4.38, "hue": 1.05, 
                "od280_od315_of_diluted_wines": 3.40, "proline": 1050.0
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )