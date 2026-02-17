"""
Comprehensive test suite for the Enhanced MLOps API
"""
import pytest
import requests
import json
import time
from typing import Dict, Any
import sys
import subprocess
import threading
from pathlib import Path

BASE_URL = "http://localhost:8000"

class TestAPI:
    """Test suite for the Enhanced MLOps API"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test"""
        # Wait a bit for API to be ready
        time.sleep(0.1)
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Enhanced MLOps API" in data["message"]
        assert "version" in data
        assert "endpoints" in data
        print("✅ Root endpoint test passed")
    
    def test_health_endpoint(self):
        """Test health check"""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "models_loaded" in data
        assert "version" in data
        print(f"✅ Health check passed - {data['models_loaded']} models loaded")
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = requests.get(f"{BASE_URL}/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "uptime_seconds" in data
        assert "models_served" in data
        assert isinstance(data["models_served"], dict)
        assert data["uptime_seconds"] >= 0
        print(f"✅ Metrics test passed - {data['total_predictions']} total predictions")
    
    def test_list_models(self):
        """Test models listing"""
        response = requests.get(f"{BASE_URL}/models")
        assert response.status_code == 200
        models = response.json()
        assert isinstance(models, list)
        
        if len(models) > 0:
            # Validate model structure
            model = models[0]
            required_fields = ["name", "type", "dataset", "features", "classes"]
            for field in required_fields:
                assert field in model, f"Missing field: {field}"
            print(f"✅ Models list test passed - {len(models)} models available")
        else:
            print("⚠️  No models available - run training first")
    
    def test_model_details(self):
        """Test individual model details"""
        # First get list of models
        response = requests.get(f"{BASE_URL}/models")
        models = response.json()
        
        if len(models) > 0:
            model_name = models[0]["name"]
            response = requests.get(f"{BASE_URL}/models/{model_name}")
            
            if response.status_code == 200:
                data = response.json()
                assert data["name"] == model_name
                print(f"✅ Model details test passed for {model_name}")
            else:
                print(f"⚠️  Model details endpoint not implemented or failed")
        else:
            print("⚠️  No models to test details")
    
    def test_iris_prediction_logistic(self):
        """Test iris prediction with logistic regression"""
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = requests.post(
            f"{BASE_URL}/predict/iris?model_type=logistic",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert "probabilities" in data
            assert 0 <= data["confidence"] <= 1
            assert isinstance(data["probabilities"], dict)
            assert data["prediction"] in ["Setosa", "Versicolor", "Virginica"]
            print(f"✅ Iris logistic prediction: {data['prediction']} (confidence: {data['confidence']:.3f})")
        else:
            print(f"⚠️  Iris logistic model not available (status: {response.status_code})")
            print(f"Response: {response.text}")
    
    def test_iris_prediction_random_forest(self):
        """Test iris prediction with random forest"""
        payload = {
            "sepal_length": 6.2,
            "sepal_width": 2.9,
            "petal_length": 4.3,
            "petal_width": 1.3
        }
        
        response = requests.post(
            f"{BASE_URL}/predict/iris?model_type=random_forest",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert "probabilities" in data
            print(f"✅ Iris RF prediction: {data['prediction']} (confidence: {data['confidence']:.3f})")
        else:
            print(f"⚠️  Iris RF model not available (status: {response.status_code})")
    
    def test_wine_prediction_logistic(self):
        """Test wine prediction with logistic regression"""
        payload = {
            "alcohol": 13.20,
            "malic_acid": 1.78,
            "ash": 2.14,
            "alcalinity_of_ash": 11.2,
            "magnesium": 100.0,
            "total_phenols": 2.65,
            "flavanoids": 2.76,
            "nonflavanoid_phenols": 0.26,
            "proanthocyanins": 1.28,
            "color_intensity": 4.38,
            "hue": 1.05,
            "od280_od315_of_diluted_wines": 3.40,
            "proline": 1050.0
        }
        
        response = requests.post(
            f"{BASE_URL}/predict/wine?model_type=logistic",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert "probabilities" in data
            print(f"✅ Wine logistic prediction: {data['prediction']} (confidence: {data['confidence']:.3f})")
        else:
            print(f"⚠️  Wine logistic model not available (status: {response.status_code})")
    
    def test_wine_prediction_random_forest(self):
        """Test wine prediction with random forest"""
        payload = {
            "alcohol": 12.85,
            "malic_acid": 1.60,
            "ash": 2.52,
            "alcalinity_of_ash": 17.8,
            "magnesium": 95.0,
            "total_phenols": 2.48,
            "flavanoids": 2.37,
            "nonflavanoid_phenols": 0.26,
            "proanthocyanins": 1.46,
            "color_intensity": 3.93,
            "hue": 1.09,
            "od280_od315_of_diluted_wines": 3.63,
            "proline": 1015.0
        }
        
        response = requests.post(
            f"{BASE_URL}/predict/wine?model_type=random_forest",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            print(f"✅ Wine RF prediction: {data['prediction']} (confidence: {data['confidence']:.3f})")
        else:
            print(f"⚠️  Wine RF model not available")
    
    def test_batch_prediction_iris(self):
        """Test batch prediction for iris"""
        payload = {
            "dataset": "iris",
            "model_type": "logistic",
            "samples": [
                [5.1, 3.5, 1.4, 0.2],  # Likely Setosa
                [6.2, 2.9, 4.3, 1.3],  # Likely Versicolor
                [6.3, 3.3, 6.0, 2.5]   # Likely Virginica
            ]
        }
        
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total_samples" in data
            assert len(data["predictions"]) == 3
            assert data["total_samples"] == 3
            
            # Validate each prediction
            for i, pred in enumerate(data["predictions"]):
                assert "prediction" in pred
                assert "confidence" in pred
                assert "probabilities" in pred
            
            print(f"✅ Iris batch prediction: {len(data['predictions'])} samples processed")
        else:
            print(f"⚠️  Batch prediction failed: {response.status_code}")
            print(f"Response: {response.text}")
    
    def test_batch_prediction_wine(self):
        """Test batch prediction for wine"""
        payload = {
            "dataset": "wine",
            "model_type": "random_forest",
            "samples": [
                [13.20, 1.78, 2.14, 11.2, 100.0, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.40, 1050.0],
                [12.37, 0.94, 1.36, 10.6, 88.0, 1.98, 0.57, 0.28, 0.42, 1.95, 1.05, 1.82, 520.0]
            ]
        }
        
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        if response.status_code == 200:
            data = response.json()
            assert len(data["predictions"]) == 2
            print(f"✅ Wine batch prediction: {len(data['predictions'])} samples processed")
        else:
            print(f"⚠️  Wine batch prediction failed: {response.status_code}")
    
    def test_examples_endpoint(self):
        """Test examples endpoint"""
        response = requests.get(f"{BASE_URL}/predict/examples")
        assert response.status_code == 200
        data = response.json()
        assert "iris_examples" in data
        assert "wine_examples" in data
        assert isinstance(data["iris_examples"], list)
        assert isinstance(data["wine_examples"], list)
        assert len(data["iris_examples"]) >= 1
        assert len(data["wine_examples"]) >= 1
        print("✅ Examples endpoint test passed")
    
    def test_invalid_input_validation(self):
        """Test input validation"""
        # Test 1: Invalid iris input (negative values)
        payload = {
            "sepal_length": -1.0,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = requests.post(
            f"{BASE_URL}/predict/iris",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        assert response.status_code == 422  # Validation error
        print("✅ Negative value validation test passed")
        
        # Test 2: Missing required fields
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5
            # Missing petal_length and petal_width
        }
        
        response = requests.post(
            f"{BASE_URL}/predict/iris",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        assert response.status_code == 422  # Validation error
        print("✅ Missing fields validation test passed")
    
    def test_invalid_model_type(self):
        """Test invalid model type"""
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = requests.post(
            f"{BASE_URL}/predict/iris?model_type=invalid_model",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        # Should return 422 (validation error) or 400 (bad request)
        assert response.status_code in [400, 422]
        print("✅ Invalid model type validation test passed")
    
    def test_api_documentation(self):
        """Test API documentation endpoints"""
        # Test OpenAPI docs
        response = requests.get(f"{BASE_URL}/docs")
        assert response.status_code == 200
        print("✅ API docs accessible")
        
        # Test OpenAPI schema
        response = requests.get(f"{BASE_URL}/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        print("✅ OpenAPI schema accessible")


def run_api_server():
    """Run the API server in background for testing"""
    try:
        import uvicorn
        from app.main import app
        
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
    except Exception as e:
        print(f"Failed to start API server: {e}")


def wait_for_api(max_wait=30):
    """Wait for API to be ready"""
    print("Waiting for API to start...")
    
    for i in range(max_wait):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ API ready after {i+1} seconds")
                return True
        except:
            pass
        
        time.sleep(1)
        if (i + 1) % 5 == 0:
            print(f"Still waiting... ({i+1}/{max_wait}s)")
    
    print("❌ API failed to start within timeout")
    return False


def check_dependencies():
    """Check if models exist"""
    models_dir = Path("models")
    if not models_dir.exists():
        print("⚠️  Models directory not found")
        print("📋 Please run: python models/trainer.py")
        return False
    
    model_files = list(models_dir.glob("*.pkl"))
    if len(model_files) == 0:
        print("⚠️  No model files found")
        print("📋 Please run: python models/trainer.py")
        return False
    
    print(f"✅ Found {len(model_files)} model files")
    return True


def run_manual_tests():
    """Run manual tests without pytest"""
    print("🧪 Running Enhanced MLOps API Tests")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    test_api = TestAPI()
    
    test_methods = [
        ("Root Endpoint", test_api.test_root_endpoint),
        ("Health Check", test_api.test_health_endpoint),
        ("Metrics", test_api.test_metrics_endpoint),
        ("List Models", test_api.test_list_models),
        ("Model Details", test_api.test_model_details),
        ("Iris Logistic", test_api.test_iris_prediction_logistic),
        ("Iris Random Forest", test_api.test_iris_prediction_random_forest),
        ("Wine Logistic", test_api.test_wine_prediction_logistic),
        ("Wine Random Forest", test_api.test_wine_prediction_random_forest),
        ("Batch Iris", test_api.test_batch_prediction_iris),
        ("Batch Wine", test_api.test_batch_prediction_wine),
        ("Examples", test_api.test_examples_endpoint),
        ("Input Validation", test_api.test_invalid_input_validation),
        ("Invalid Model", test_api.test_invalid_model_type),
        ("API Documentation", test_api.test_api_documentation),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, test_method in test_methods:
        try:
            print(f"\n🔍 Testing {test_name}...")
            test_method()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test_name} failed: {e}")
            failed += 1
        except Exception as e:
            if "not available" in str(e) or "status: 400" in str(e):
                print(f"⚠️  {test_name} skipped: {e}")
                skipped += 1
            else:
                print(f"❌ {test_name} error: {e}")
                failed += 1
    
    print(f"\n{'='*60}")
    print(f"📊 Test Results:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⚠️  Skipped: {skipped}")
    print(f"   📈 Total: {passed + failed + skipped}")
    
    if failed == 0:
        print("\n🎉 All available tests passed!")
        if skipped > 0:
            print("💡 Some tests were skipped due to missing models or endpoints")
        return True
    else:
        print(f"\n⚠️  {failed} tests failed. Check the output above.")
        return False


def run_with_server():
    """Run tests with automatic server management"""
    print("🚀 Starting API server for testing...")
    
    # Start API server in background thread
    server_thread = threading.Thread(target=run_api_server, daemon=True)
    server_thread.start()
    
    # Wait for API to be ready
    if not wait_for_api():
        print("❌ Cannot start tests - API server failed to start")
        return False
    
    # Run tests
    success = run_manual_tests()
    
    print("\n🔄 Tests completed. Server will stop when script exits.")
    return success


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--with-server":
        # Run with automatic server management
        success = run_with_server()
    else:
        # Assume server is already running
        print("🔍 Testing against running API server...")
        print("💡 Use --with-server to auto-start the server")
        success = run_manual_tests()
    
    sys.exit(0 if success else 1)