"""
Configuration File
Central place for all project settings and constants
"""

# Data paths
DATA_RAW_PATH = 'data/raw/student-mat.csv'
DATA_PROCESSED_PATH = 'data/processed/student_processed.csv'

# Model paths
MODEL_PATH = 'models/performance_model.pkl'
MODEL_METRICS_PATH = 'models/model_metrics.json'

# Dataset configuration
TARGET_COLUMN = 'G3'  # Final grade (0-20)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model parameters
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 15,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Dataset download URL
DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/student/student-mat.csv'

# Streamlit UI configuration
APP_TITLE = "🎓 AI Student Performance Predictor"
APP_DESCRIPTION = "Predict student performance and get personalized learning recommendations"

print("✓ Configuration loaded successfully")
