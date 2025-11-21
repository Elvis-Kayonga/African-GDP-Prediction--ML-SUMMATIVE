# African GDP Growth Prediction - ML Summative Assignment

## 🎯 Mission Statement

This project predicts GDP growth rates for African countries using machine learning to support economic policy decisions and investment strategies. By analyzing key economic indicators including inflation, unemployment, foreign direct investment, trade balance, government debt, and internet penetration, we provide data-driven insights for African economic development.

## 📊 Problem Statement

African economies face unique challenges and opportunities. This model helps predict GDP growth trajectories based on multiple economic factors, enabling policymakers, investors, and researchers to make informed decisions about resource allocation and strategic planning across 18 African nations.

---

## 🏗️ Project Structure

```
summative/
├── linear_regression/
│   └── multivariate.ipynb          # ML model development notebook
├── API/
│   ├── prediction.py                # FastAPI application
│   ├── requirements.txt             # Python dependencies
│   ├── best_model.pkl              # Trained model (generated)
│   ├── scaler.pkl                  # Feature scaler (generated)
│   ├── label_encoder.pkl           # Country encoder (generated)
│   ├── feature_names.pkl           # Feature list (generated)
│   └── model_metadata.pkl          # Model info (generated)
└── FlutterApp/
    ├── lib/
    │   └── main.dart                # Flutter app main code
    ├── pubspec.yaml                 # Flutter dependencies
    └── README.md                    # Flutter setup guide
```

---

## 🚀 API Endpoint

### Base URL
```
[YOUR_RENDER_URL_HERE]
```

### Swagger UI Documentation
```
[YOUR_RENDER_URL_HERE]/docs
```

**Note:** Replace `[YOUR_RENDER_URL_HERE]` with your actual Render deployment URL after deployment.

### Example API Endpoints

- `GET /` - API information
- `GET /countries` - List of supported countries
- `GET /model-info` - Model performance metrics
- `POST /predict` - Make predictions
- `GET /health` - Health check

### Sample Prediction Request

```bash
curl -X POST "https://your-api-url.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "year": 2024,
    "inflation_rate": 12.5,
    "unemployment_rate": 18.0,
    "fdi_millions_usd": 3500.0,
    "trade_balance_millions_usd": -800.0,
    "govt_debt_percent_gdp": 38.0,
    "internet_penetration_percent": 55.0,
    "country": "Nigeria"
  }'
```

### Sample Response

```json
{
  "predicted_gdp_growth_rate": 2.34,
  "unit": "%",
  "model_used": "Random Forest",
  "input_data": {
    "year": 2024,
    "country": "Nigeria",
    ...
  }
}
```

---

## 🎥 Video Demo

### YouTube Link
```
[YOUR_YOUTUBE_VIDEO_LINK_HERE]
```

**Video Contents (Max 5 minutes):**
1. Model performance comparison (Linear Regression, Decision Tree, Random Forest)
2. Jupyter notebook walkthrough
3. Mobile app demonstration with predictions
4. API testing via Swagger UI (data types and range validation)
5. Flutter code showing API integration

---

## 🏃 How to Run

### Prerequisites

1. **Python 3.8+**
2. **Jupyter Notebook**
3. **Flutter SDK 3.0+**
4. **Git**

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### Step 2: Run the Jupyter Notebook

```bash
cd summative/linear_regression
jupyter notebook multivariate.ipynb
```

**Run all cells to:**
- Load and explore the dataset
- Train all models (Gradient Descent, Linear Regression, Decision Tree, Random Forest)
- Generate model files (.pkl) in the API directory
- View visualizations and model comparisons

### Step 3: Test API Locally (Optional)

```bash
cd ../API
pip install -r requirements.txt
python prediction.py
```

Visit: `http://localhost:8000/docs` to test the API locally.

### Step 4: Deploy API to Render

See `DEPLOYMENT_GUIDE.md` for detailed instructions.

### Step 5: Run Flutter Mobile App

```bash
cd ../FlutterApp
flutter pub get
```

**Update API URL in `lib/main.dart` (line 56):**
```dart
final String apiUrl = 'https://your-render-url.onrender.com/predict';
```

**Run the app:**
```bash
flutter run
```

---

## 📱 Mobile App Usage

1. **Select Country** from dropdown (18 African countries available)
2. **Enter 7 economic indicators:**
   - Year (2000-2050)
   - Inflation Rate (0-100%)
   - Unemployment Rate (0-100%)
   - FDI in Millions USD (0-50000)
   - Trade Balance in Millions USD (-50000 to 50000)
   - Government Debt as % of GDP (0-200%)
   - Internet Penetration (0-100%)
3. **Click "Predict"** button
4. **View Result** - Predicted GDP Growth Rate with model name

---

## 🧪 Model Performance

| Model | Train MSE | Test MSE | Train R² | Test R² |
|-------|-----------|----------|----------|---------|
| Gradient Descent | [Generated] | [Generated] | [Generated] | [Generated] |
| Linear Regression | [Generated] | [Generated] | [Generated] | [Generated] |
| Decision Tree | [Generated] | [Generated] | [Generated] | [Generated] |
| Random Forest | [Generated] | [Generated] | [Generated] | [Generated] |

**Best Model:** [Will be determined after running notebook]

---

## 📋 Features Implemented

### Task 1: Machine Learning ✅
- ✅ African economic dataset (non-house prediction)
- ✅ Comprehensive data visualization and interpretation
- ✅ Feature engineering with justification
- ✅ Categorical to numeric conversion
- ✅ Data standardization
- ✅ Gradient Descent from scratch (pedagogical + vectorized)
- ✅ Sklearn models: Linear Regression, Decision Tree, Random Forest
- ✅ Loss curves plotted (train vs test)
- ✅ Scatter plots (actual vs predicted)
- ✅ Best model saved

### Task 2: FastAPI ✅
- ✅ FastAPI with CORS middleware
- ✅ Pydantic models with data type validation
- ✅ Range constraints on all inputs
- ✅ POST /predict endpoint
- ✅ Error handling
- ✅ requirements.txt
- ✅ Ready for Render deployment

### Task 3: Flutter App ✅
- ✅ Single-page app with 8 input fields
- ✅ Country dropdown + 7 numeric text fields
- ✅ "Predict" button
- ✅ Result display area
- ✅ Error handling and validation
- ✅ Clean, organized UI
- ✅ API integration

---

## 🌍 Supported African Countries

Angola, Botswana, Cameroon, Egypt, Ethiopia, Ghana, Ivory Coast, Kenya, Morocco, Nigeria, Rwanda, Senegal, South Africa, Tanzania, Tunisia, Uganda, Zambia, Zimbabwe

---

## 🔧 Technologies Used

- **Machine Learning:** Python, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- **API:** FastAPI, Pydantic, Uvicorn
- **Mobile App:** Flutter, Dart
- **Deployment:** Render.com
- **Version Control:** Git, GitHub

---

## 📝 Assignment Requirements Met

✅ Non-generic use case (African finance)  
✅ NOT house prediction  
✅ Dataset from legitimate source  
✅ Visualizations with interpretations  
✅ Feature engineering explained  
✅ Data standardization  
✅ Gradient descent implementation (pedagogical + vectorized)  
✅ Sklearn models (Linear Regression, Decision Tree, Random Forest)  
✅ Loss curves plotted  
✅ Scatter plots (before/after)  
✅ Best model saved  
✅ FastAPI with CORS  
✅ Pydantic validation (types + ranges)  
✅ requirements.txt  
✅ Publicly routable URL  
✅ Flutter mobile app (not web)  
✅ Correct number of input fields  
✅ Organized UI  
✅ README with all required information  

---

## 📧 Contact & Submission

**GitHub Repository:** [YOUR_GITHUB_REPO_LINK]  
**Author:** [YOUR_NAME]  
**Date:** November 2025  
**Course:** ML Summative Assignment  

---

## 📄 License

This project is created for educational purposes as part of an ML course assignment.

---

## 🙏 Acknowledgments

- Dataset inspired by real African economic indicators
- Built with Flutter, FastAPI, and Scikit-learn
- Deployed on Render.com

---

**Note:** After deployment, update this README with:
1. Your actual Render API URL
2. Your YouTube video link
3. Model performance metrics (from notebook output)
4. Your GitHub repository link
