"""
YugenAI MLOps Dashboard - Streamlit Application
"""
import streamlit as st
from pathlib import Path
import sys
import os
import json
from datetime import datetime

# Add the current directory to the Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Import configuration
from config import PAGE_CONFIG, UI, MODEL_CONFIG, CLASS_NAMES

# Import database utilities
from utils.db import log_request, update_request

# Set page configuration
st.set_page_config(**PAGE_CONFIG)

# Custom CSS
st.markdown(
    f"""
    <style>
    .main .block-container {{
        max-width: {UI['max_width']}px;
        padding: 2rem 1rem;
    }}
    .stApp {{
        max-width: 100%;
    }}
    .stSidebar {{
        width: 25%;
        min-width: 250px;
        max-width: 300px;
    }}
    .prediction-box {{
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }}
    .stButton>button {{
        width: 100%;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }}
    .stSlider [data-baseweb="slider"] {{
        margin-top: 1.5rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    """Main application function."""
    # Sidebar
    st.sidebar.title("MLOps Platform")
    st.sidebar.markdown("### Machine Learning Workflow")
    
    # Navigation
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "📊 Data Preparation", "🤖 Model Training", "🌺 Iris Classifier", "📈 Model Monitoring"],
        label_visibility="collapsed"
    )
    
    # Display the selected page
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Data Preparation":
        from pages.data_preparation import main as data_prep
        data_prep()
    elif page == "🤖 Model Training":
        from pages.model_training import main as model_train
        model_train()
    elif page == "🌺 Iris Classifier":
        from pages.iris_classifier import main as iris_classifier
        iris_classifier()
    elif page == "📈 Model Monitoring":
        show_model_monitoring()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **MLOps Platform**  
        Version 1.0.0  
        
        [GitHub Repository](https://github.com/yourusername/mlops-platform)
        """
    )

def show_home():
    """Display the home page with project overview and navigation."""
    st.title("Welcome to MLOps Platform")
    
    # Header with description
    st.markdown("""
    A comprehensive platform for managing your end-to-end machine learning workflow,
    from data preparation to model deployment and monitoring.
    """)
    
    # Features section
    st.markdown("## 🚀 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Data Preparation")
        st.markdown("""
        - Upload and manage datasets
        - Handle missing values and outliers
        - Feature engineering and transformation
        - Save processed data for training
        """)
        
        st.markdown("### 🤖 Model Training")
        st.markdown("""
        - Train multiple ML models
        - Hyperparameter tuning
        - Cross-validation
        - Save trained models
        """)
    
    with col2:
        st.markdown("### 🌺 Iris Classifier")
        st.markdown("""
        - Interactive classification demo
        - Real-time predictions
        - Probability estimates
        - Model explainability
        """)
        
        st.markdown("### 📈 Model Monitoring")
        st.markdown("""
        - Track model performance
        - Monitor data drift
        - Alert on anomalies
        - Model versioning
        """)
    
    # Getting Started section
    st.markdown("## 🏁 Getting Started")
    st.markdown("""
    1. **Prepare your data** using the Data Preparation tool
    2. **Train models** with your prepared data
    3. **Test predictions** using the Iris Classifier
    4. **Monitor** your models in production
    
    Select any tool from the sidebar to get started!
    """)

def show_model_monitoring():
    """Display the model monitoring page."""
    st.title("Model Monitoring")
    st.info("Model monitoring features will be available in a future update.")

if __name__ == "__main__":
    main()
