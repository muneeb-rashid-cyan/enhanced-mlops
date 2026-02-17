"""
Pydantic schemas for API request/response models
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class DatasetType(str, Enum):
    """Available datasets"""
    iris = "iris"
    wine = "wine"

class ModelType(str, Enum):
    """Available model types"""
    logistic = "logistic"
    random_forest = "random_forest"

# Input schemas
class IrisInput(BaseModel):
    """Iris flower classification input"""
    sepal_length: float = Field(..., ge=0, le=10, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, le=10, description="Sepal width in cm") 
    petal_length: float = Field(..., ge=0, le=10, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, le=10, description="Petal width in cm")

class WineInput(BaseModel):
    """Wine classification input"""
    alcohol: float = Field(..., ge=0, le=20)
    malic_acid: float = Field(..., ge=0, le=10)
    ash: float = Field(..., ge=0, le=5)
    alcalinity_of_ash: float = Field(..., ge=0, le=50)
    magnesium: float = Field(..., ge=0, le=200)
    total_phenols: float = Field(..., ge=0, le=5)
    flavanoids: float = Field(..., ge=0, le=10)
    nonflavanoid_phenols: float = Field(..., ge=0, le=2)
    proanthocyanins: float = Field(..., ge=0, le=5)
    color_intensity: float = Field(..., ge=0, le=15)
    hue: float = Field(..., ge=0, le=5)
    od280_od315_of_diluted_wines: float = Field(..., ge=0, le=10)
    proline: float = Field(..., ge=0, le=2000)

class BatchPredictionInput(BaseModel):
    """Batch prediction input"""
    dataset: DatasetType
    model_type: ModelType = ModelType.logistic
    samples: List[List[float]] = Field(..., min_items=1, max_items=100)

# Output schemas
class PredictionOutput(BaseModel):
    """Single prediction output"""
    prediction: str = Field(..., description="Predicted class name")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")

class BatchPredictionOutput(BaseModel):
    """Batch prediction output"""
    predictions: List[PredictionOutput]
    model_info: Dict[str, Any]
    total_samples: int

class ModelInfo(BaseModel):
    """Model information"""
    name: str
    type: str
    dataset: str
    accuracy: Optional[float] = None
    features: List[str]
    classes: List[str]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    timestamp: str
    models_loaded: int
    version: str

class MetricsResponse(BaseModel):
    """Metrics response"""
    total_predictions: int
    models_served: Dict[str, int]
    average_response_time: float
    uptime_seconds: float