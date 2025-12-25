# 🎓 AI-based Student Performance Predictor with Learning Recommendations

A machine learning project that predicts student performance and recommends personalized learning resources.

## 📁 Project Structure

```
minor project/
├── data/
│   ├── raw/              # Original UCI dataset (student-mat.csv)
│   └── processed/        # Cleaned and prepared data
├── models/               # Trained ML models
├── src/                  # Source code modules
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data cleaning & preparation
│   ├── model_training.py       # ML model training & evaluation
│   ├── recommendation.py       # Learning resource recommendations
│   └── utils.py                # Helper functions
├── app/
│   └── streamlit_app.py   # Web UI application
├── notebooks/            # Jupyter notebooks for exploration
├── config.py             # Central configuration file
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── .gitignore           # Git ignore rules
```

## 📄 File Descriptions

### **src/ (Source Code)**
- **data_preprocessing.py**: Loads CSV, removes missing values, prepares features for ML
- **model_training.py**: Trains Random Forest model, evaluates performance (MAE, RMSE, R²)
- **recommendation.py**: Generates personalized learning resources based on predicted scores
- **utils.py**: Helper functions for file operations and configuration

### **Core Files**
- **config.py**: Central settings (file paths, model parameters, dataset URL)
- **app/streamlit_app.py**: Interactive web interface for predictions and recommendations

### **Data**
- **data/raw/**: Stores original UCI student-mat.csv dataset
- **data/processed/**: Stores cleaned data after preprocessing

### **Models**
- **models/**: Stores trained model and metrics

## 🚀 Next Steps

1. ✅ Project structure created
2. 📥 Download UCI dataset
3. 🔧 Build data preprocessing
4. 🤖 Train ML model
5. 💡 Create recommendation engine
6. 🎨 Build Streamlit UI
7. 📚 Add documentation
8. ✨ Final testing

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app/streamlit_app.py
```

## Dataset

- Source: UCI Machine Learning Repository — Student Performance (Mathematics).
- Size: 395 instances; mixed categorical and numerical features.
- Feature groups: demographics (e.g., age, sex), prior grades (G1, G2), study/engagement (study time, failures, absences), family and social context.
- Target: `G3` — final grade on a 0–20 scale.
- File: data/raw/student-mat.csv.

---

**Status**: Step 1 Complete ✓
