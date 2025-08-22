"""
YugenAI MLOps Dashboard - Streamlit Application
"""
import streamlit as st
from pathlib import Path
import sys

# Add the current directory to the Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Set page configuration
st.set_page_config(
    page_title="Home",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide the default Streamlit menu and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Main content
st.title("Welcome to MLOps Platform")
st.write("## End-to-End Machine Learning Operations")

st.write("### Our MLOps Process:")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info("1. Data Preparation")
    st.write("Clean, preprocess and prepare your data for model training.")

with col2:
    st.info("2. Model Training")
    st.write("Train and evaluate machine learning models with your prepared data.")

with col3:
    st.info("3. Model Inference")
    st.write("Make predictions using trained models. Try our House Value Prediction or Iris Classification.")

with col4:
    st.info("4. Monitoring")
    st.write("Track and monitor model performance and requests through our logs.")

st.write("---")
st.write("### Getting Started")
st.write("Please use the page navigation in your browser to access different sections of the MLOps platform.")
