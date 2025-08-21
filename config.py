"""
Configuration settings for the YugenAI Streamlit application.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Model Configuration
MODEL_CONFIG = {
    "iris": {
        "name": "Iris Flower Classifier",
        "description": "Classify iris flowers into setosa, versicolor, or virginica",
        "features": {
            "sepal_length": {"label": "Sepal Length (cm)", "min": 4.0, "max": 8.0, "step": 0.1, "value": 5.8},
            "sepal_width": {"label": "Sepal Width (cm)", "min": 2.0, "max": 4.5, "step": 0.1, "value": 3.0},
            "petal_length": {"label": "Petal Length (cm)", "min": 1.0, "max": 7.0, "step": 0.1, "value": 3.8},
            "petal_width": {"label": "Petal Width (cm)", "min": 0.1, "max": 2.5, "step": 0.1, "value": 1.2}
        },
        "endpoint": "/predict_iris"
    },
    "housing": {
        "name": "Housing Price Predictor",
        "description": "Predict housing prices based on property features",
        "endpoint": "/predict_housing"
    }
}

# Class names for display
CLASS_NAMES = {
    "Iris-setosa": {"display": "Setosa", "color": "#1f77b4"},
    "Iris-versicolor": {"display": "Versicolor", "color": "#2ca02c"},
    "Iris-virginica": {"display": "Virginica", "color": "#d62728"}
}

# Page configuration
PAGE_CONFIG = {
    "page_title": "YugenAI MLOps Dashboard",
    "page_icon": "🌺",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# UI Constants
UI = {
    "max_width": 1200,
    "sidebar_width": 0.2
}

# Logging configuration
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "app.log"
}
