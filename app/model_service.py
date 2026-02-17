"""
Model management service - FIXED VERSION
"""
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from app.schemas import ModelInfo

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages loading and inference for multiple models"""
    
    def __init__(self):
        self.models: Dict[str, Dict] = {}
        self.model_stats: Dict[str, int] = {}
        self._load_available_models()
    
    def _load_available_models(self):
        """Load all available models from the models directory"""
        models_dir = Path("models")
        if not models_dir.exists():
            logger.warning("Models directory not found. Run training first.")
            return
        
        model_files = list(models_dir.glob("*.pkl"))
        logger.info(f"Found {len(model_files)} model files")
        
        for model_file in model_files:
            try:
                model_data = joblib.load(model_file)
                model_key = model_file.stem  # filename without extension
                
                self.models[model_key] = {
                    'model': model_data['model'],
                    'target_names': model_data['target_names'],
                    'feature_names': model_data['feature_names'],
                    'path': str(model_file)
                }
                self.model_stats[model_key] = 0
                logger.info(f"Loaded model: {model_key}")
                
            except Exception as e:
                logger.error(f"Failed to load {model_file}: {e}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available model keys"""
        return list(self.models.keys())
    
    def get_model_info(self, model_key: str) -> Optional[ModelInfo]:
        """Get information about a specific model"""
        if model_key not in self.models:
            return None
        
        model_data = self.models[model_key]
        
        # Parse model name: iris_logistic_model -> dataset=iris, type=logistic
        if model_key.endswith('_model'):
            model_name_clean = model_key[:-6]  # Remove '_model'
        else:
            model_name_clean = model_key
        
        parts = model_name_clean.split('_')
        if len(parts) >= 2:
            dataset = parts[0]  # iris or wine
            model_type = '_'.join(parts[1:])  # logistic or random_forest
        else:
            dataset, model_type = model_name_clean, "unknown"
        
        return ModelInfo(
            name=model_key,
            type=model_type,
            dataset=dataset,
            features=model_data['feature_names'].tolist() if hasattr(model_data['feature_names'], 'tolist') else model_data['feature_names'],
            classes=model_data['target_names'].tolist() if hasattr(model_data['target_names'], 'tolist') else model_data['target_names']
        )
    
    def predict(self, model_key: str, features: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Make prediction with specified model"""
        
        # 🔧 FIX: Convert API model key to actual file model key
        # API calls with: iris_logistic
        # File names are: iris_logistic_model
        actual_model_key = f"{model_key}_model"
        
        logger.info(f"API requested model: {model_key}")
        logger.info(f"Looking for file model: {actual_model_key}")
        logger.info(f"Available models: {list(self.models.keys())}")
        
        if actual_model_key not in self.models:
            raise ValueError(f"Model {model_key} not found. Available: {list(self.models.keys())}")
        
        model_data = self.models[actual_model_key]
        model = model_data['model']
        target_names = model_data['target_names']
        
        # Make prediction
        prediction_idx = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get class name and confidence
        prediction_name = target_names[prediction_idx]
        confidence = float(probabilities[prediction_idx])
        
        # Create probabilities dict
        prob_dict = {
            target_names[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        # Update stats for the actual model key
        self.model_stats[actual_model_key] += 1
        
        return prediction_name, confidence, prob_dict
    
    def batch_predict(self, model_key: str, features_list: List[List[float]]) -> List[Tuple[str, float, Dict[str, float]]]:
        """Make batch predictions"""
        
        # 🔧 FIX: Same conversion for batch predictions
        actual_model_key = f"{model_key}_model"
        
        if actual_model_key not in self.models:
            raise ValueError(f"Model {model_key} not found")
        
        model_data = self.models[actual_model_key]
        model = model_data['model']
        target_names = model_data['target_names']
        
        # Convert to numpy array
        features_array = np.array(features_list)
        
        # Make batch predictions
        predictions = model.predict(features_array)
        probabilities = model.predict_proba(features_array)
        
        results = []
        for i, (pred_idx, probs) in enumerate(zip(predictions, probabilities)):
            prediction_name = target_names[pred_idx]
            confidence = float(probs[pred_idx])
            prob_dict = {
                target_names[j]: float(prob)
                for j, prob in enumerate(probs)
            }
            results.append((prediction_name, confidence, prob_dict))
        
        # Update stats
        self.model_stats[actual_model_key] += len(features_list)
        
        return results
    
    def get_stats(self) -> Dict[str, int]:
        """Get model usage statistics"""
        return self.model_stats.copy()
    
    def reload_models(self):
        """Reload all models (useful for model updates)"""
        logger.info("Reloading models...")
        self.models.clear()
        self.model_stats.clear()
        self._load_available_models()
        