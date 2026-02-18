# Enhanced MLOps Platform with Azure Deployment

A complete end-to-end MLOps pipeline demonstrating CI/CD automation, model training, and cloud deployment on Azure Web App. This project showcases production-ready machine learning workflows with automated testing, deployment, and API serving.

## 🎯 Project Overview

This project implements a comprehensive MLOps solution that trains multiple machine learning models (Random Forest, Logistic Regression, and SVM), exposes them through a FastAPI REST API, and automatically deploys to Azure Web App using GitHub Actions CI/CD pipeline.

**Key Features:**
- Automated ML model training pipeline
- RESTful API with FastAPI for model inference
- Comprehensive test suite with pytest
- CI/CD automation with GitHub Actions
- Production deployment on Azure Web App
- Interactive API documentation with Swagger UI
- Model versioning and management with MLflow

## 🏗️ Architecture

```
├── app/
│   └── main.py                 # FastAPI application
├── models/
│   └── trainer.py              # ML model training script
├── tests/
│   └── test_api.py            # API test suite
├── .github/
│   └── workflows/
│       └── azure-pipelines.yml # CI/CD pipeline configuration
├── requirements.txt            # Python dependencies
└── README.md
```

## 🚀 Features

### Machine Learning Models
- **Random Forest Classifier** - Ensemble learning for robust predictions
- **Logistic Regression** - Linear model for baseline performance
- **Support Vector Machine (SVM)** - Non-linear classification

### API Endpoints
- `GET /` - Welcome message and API information
- `GET /health` - Health check endpoint
- `GET /models` - List all available trained models
- `POST /predict/{model_name}` - Make predictions using specified model
- `GET /docs` - Interactive Swagger UI documentation
- `GET /redoc` - Alternative API documentation

### CI/CD Pipeline
The GitHub Actions workflow automatically:
1. **Tests** - Runs comprehensive test suite
2. **Trains Models** - Trains all ML models
3. **Validates** - Checks API endpoints
4. **Deploys** - Pushes to Azure Web App
5. **Verifies** - Confirms deployment success

## 📋 Prerequisites

- Python 3.11+
- Azure Account with active subscription
- GitHub Account
- Azure CLI (for local development)
- Git

## 🔧 Local Development Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd enhanced-mlops-platform
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train Models Locally
```bash
python models/trainer.py
```

### 5. Run the API Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Test the API
```bash
# Run test suite
python tests/test_api.py

# Or use pytest
pytest tests/

# Manual testing with curl
curl http://localhost:8000/health
curl http://localhost:8000/models
```

### 7. Access API Documentation
Open your browser and navigate to:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## ☁️ Azure Deployment

### Architecture Overview
The application is deployed on **Azure Web App** (Platform as a Service) which provides:
- Automatic scaling
- Built-in load balancing
- SSL/TLS certificates
- Continuous deployment integration
- Application monitoring

### Deployment Process

#### Step 1: Create Azure Resources

1. **Create Resource Group**
```bash
az group create --name RG_NAME --location RG_LOCATION
```

2. **Create App Service Plan**
```bash
az appservice plan create \
  --name mlops-service-plan \
  --resource-group RG_NAME \
  --sku B1 \
  --is-linux
```

3. **Create Web App**
```bash
az webapp create \
  --name enhanced-mlops-api-muneeb \
  --resource-group RG_NAME \
  --plan mlops-service-plan \
  --runtime "PYTHON:3.11"
```

#### Step 2: Configure GitHub Secrets

Set up the following secrets in your GitHub repository (Settings → Secrets and variables → Actions):

| Secret Name | Description | How to Get |
|-------------|-------------|------------|
| `AZURE_WEBAPP_NAME` | Your Azure Web App name | `enhanced-mlops-api-muneeb` |
| `AZURE_USERNAME` | Azure account email | Your Azure login email |
| `AZURE_PASSWORD` | Azure account password | Your Azure login password |
| `AZURE_TENANT_ID` | Azure tenant ID | `az account show --query tenantId -o tsv` |
| `AZURE_SUBSCRIPTION_ID` | Azure subscription ID | `az account show --query id -o tsv` |

#### Step 3: Enable Basic Authentication

In Azure Portal:
1. Navigate to your Web App → **Configuration** → **General settings**
2. Enable **SCM Basic Auth Publishing Credentials**
3. Enable **FTP Basic Auth Publishing Credentials**
4. Click **Save**

#### Step 4: Deploy via GitHub Actions

The deployment happens automatically when you push to the `main` branch:

```bash
git add .
git commit -m "Deploy to Azure"
git push origin main
```

The GitHub Actions workflow will:
1. ✅ Run tests
2. ✅ Train models
3. ✅ Validate API
4. ✅ Deploy to Azure
5. ✅ Set startup command
6. ✅ Verify deployment

#### Step 5: Verify Deployment

Once deployed, access your live API at:
- **Base URL**: https://enhanced-mlops-api-muneeb.azurewebsites.net
- **Health Check**: https://enhanced-mlops-api-muneeb.azurewebsites.net/health
- **API Docs**: https://enhanced-mlops-api-muneeb.azurewebsites.net/docs
- **Models List**: https://enhanced-mlops-api-muneeb.azurewebsites.net/models

### Making Predictions

**Example Request:**
```bash
curl -X POST "https://enhanced-mlops-api-muneeb.azurewebsites.net/predict/random_forest" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

**Example Response:**
```json
{
  "model": "random_forest",
  "prediction": "setosa",
  "probability": 0.98,
  "timestamp": "2026-02-17T14:30:00Z"
}
```

## 🧪 Testing

The project includes comprehensive testing:

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py

# Run with coverage
pytest tests/ --cov=app --cov=models
```

## 📊 Monitoring and Logs

### Azure Portal
1. Navigate to your Web App → **Monitoring** → **Log stream**
2. View real-time application logs
3. Check deployment history under **Deployment Center**

## 🔒 Security Best Practices

- ✅ Secrets stored in GitHub Secrets (never in code)
- ✅ HTTPS enforced on Azure Web App
- ✅ Environment variables for configuration
- ✅ Input validation on API endpoints
- ✅ Rate limiting (can be configured in Azure)
- ✅ Regular dependency updates

## 📈 Performance Optimization

- **App Service Plan**: Using B1 (Basic) tier for cost-effective production
- **Auto-scaling**: Can be enabled for higher tiers
- **Caching**: Models loaded once at startup
- **Async Operations**: FastAPI's async capabilities utilized
- **Compression**: Enabled by default on Azure Web App

## 🛠️ Troubleshooting

### Common Issues and Solutions

**Issue: Deployment fails with 403 error**
- **Solution**: Enable Basic Auth in Azure Portal → Configuration → General settings

**Issue: MFA blocking authentication**
- **Solution**: Use Service Principal instead of username/password authentication

**Issue: Models not found after deployment**
- **Solution**: Ensure `models/trainer.py` runs in CI/CD pipeline before deployment

**Issue: API returns 500 error**
- **Solution**: Check Azure Web App logs via Portal → Log stream

**Issue: Startup command not working**
- **Solution**: Verify startup command in Configuration → General settings:
  ```
  uvicorn app.main:app --host 0.0.0.0 --port 8000
  ```

## 🎓 Learning Outcomes

### Technical Skills Demonstrated
- ✅ **MLOps Pipeline Design** - End-to-end automated ML workflow
- ✅ **CI/CD Implementation** - GitHub Actions for automated deployment
- ✅ **Cloud Deployment** - Azure Web App configuration and management
- ✅ **API Development** - RESTful API design with FastAPI
- ✅ **Testing** - Comprehensive test suite with pytest
- ✅ **Version Control** - Git workflow with feature branches
- ✅ **Documentation** - Complete project documentation
- ✅ **Troubleshooting** - Resolving authentication, dependency, and deployment issues

### Key Challenges Overcome
1. **Dependency Management** - Resolved conflicts with azureml-mlflow and ruamel.yaml through version pinning
2. **Authentication** - Navigated MFA requirements and basic auth configuration
3. **Deployment Automation** - Built complete CI/CD pipeline with Azure CLI integration
4. **Error Handling** - Implemented robust error handling and logging
5. **Resource Management** - Optimized Azure resource usage for cost-effectiveness

## 📚 Technologies Used

| Category | Technologies |
|----------|-------------|
| **ML Framework** | scikit-learn, MLflow, joblib |
| **API Framework** | FastAPI, Uvicorn, Pydantic |
| **Testing** | pytest, httpx, requests |
| **Cloud Platform** | Azure Web App, Azure CLI |
| **CI/CD** | GitHub Actions |
| **Languages** | Python 3.11 |
| **Version Control** | Git, GitHub |

## 📞 API Documentation

Full interactive API documentation is available at:
- **Swagger UI**: https://enhanced-mlops-api-muneeb.azurewebsites.net/docs
- **ReDoc**: https://enhanced-mlops-api-muneeb.azurewebsites.net/redoc

## 🔄 Future Enhancements

- [ ] Model versioning with MLflow Model Registry
- [ ] A/B testing framework
- [ ] Model performance monitoring
- [ ] Automated model retraining pipeline
- [ ] Database integration for prediction logging
- [ ] Advanced authentication (OAuth2, API keys)
- [ ] Kubernetes deployment (AKS)
- [ ] Model explainability features (SHAP, LIME)
- [ ] Real-time prediction streaming
- [ ] Multi-model ensemble predictions

## 📝 License

This project is created for educational and portfolio purposes.

## 👤 Author

**Muneeb Rashid**
- Email: muneeb Rashid
- Project Duration: February 2026


## 🙏 Acknowledgments

- Anthropic Claude - For technical assistance and troubleshooting
- Azure Documentation - For deployment references
- FastAPI Community - For excellent framework documentation

---

**Project Status**: ✅ Production Deployed | **Last Updated**: February 17, 2026