"""
Model training and MLflow logging
"""
import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Simple model trainer with MLflow integration"""
    
    def __init__(self):
        # Set MLflow tracking to local file storage
        mlflow.set_tracking_uri("file:./mlruns")
        
    def load_dataset(self, dataset_name: str = "iris"):
        """Load dataset (iris or wine)"""
        if dataset_name == "iris":
            data = load_iris()
            target_names = ["Setosa", "Versicolor", "Virginica"]
        elif dataset_name == "wine":
            data = load_wine()
            target_names = ["Class_0", "Class_1", "Class_2"]
        else:
            raise ValueError("Dataset must be 'iris' or 'wine'")
            
        return data.data, data.target, data.feature_names, target_names
        
    def train_model(self, model_type: str = "logistic", dataset: str = "iris"):
        """Train model with MLflow logging"""
        
        experiment_name = f"{dataset}_classification"
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run():
            # Load data
            X, y, feature_names, target_names = self.load_dataset(dataset)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Log dataset info
            mlflow.log_param("dataset", dataset)
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("n_samples", X.shape[0])
            mlflow.log_param("model_type", model_type)
            
            # Create model with better parameters
            if model_type == "logistic":
                model = LogisticRegression(random_state=42, max_iter=2000, solver='lbfgs')  # ✅ Fixed convergence
            elif model_type == "random_forest":
                model = RandomForestClassifier(random_state=42, n_estimators=100)
            else:
                raise ValueError("Model type must be 'logistic' or 'random_forest'")
            
            # Train model
            logger.info(f"Training {model_type} model on {dataset} dataset...")
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("cv_mean", cv_scores.mean())
            mlflow.log_metric("cv_std", cv_scores.std())
            
            # ✅ Fixed MLflow model logging (no slashes in name)
            model_name = f"{dataset}_{model_type}_classifier"  # ✅ No slashes
            mlflow.sklearn.log_model(
                model, 
                "model",  # ✅ Simple artifact path
                registered_model_name=model_name
            )
            
            # Save locally for API (create directory first)
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            local_path = models_dir / f"{dataset}_{model_type}_model.pkl"
            joblib.dump({
                'model': model,
                'target_names': target_names,
                'feature_names': feature_names
            }, local_path)
            
            logger.info(f"Model saved: {local_path}")
            logger.info(f"Accuracy: {accuracy:.3f}")
            logger.info(f"CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            return str(local_path), accuracy

def main():
    """Train multiple models"""
    trainer = ModelTrainer()
    
    # Train different model combinations
    combinations = [
        ("iris", "logistic"),
        ("iris", "random_forest"),
        ("wine", "logistic"),
        ("wine", "random_forest")
    ]
    
    results = {}
    for dataset, model_type in combinations:
        try:
            model_path, accuracy = trainer.train_model(model_type, dataset)
            results[f"{dataset}_{model_type}"] = {
                "path": model_path,
                "accuracy": accuracy
            }
            print(f"✅ {dataset} {model_type}: {accuracy:.3f}")
        except Exception as e:
            print(f"❌ {dataset} {model_type}: {e}")
    
    print(f"\n🎉 Training completed! Models: {len(results)}")
    print("\n📋 View experiments:")
    print("mlflow ui --port 5000")
    
    return results

if __name__ == "__main__":
    main()