"""
Housing Price Prediction Page
"""
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import os

# Import components
from components.feature_inputs import get_housing_features
from components.results_display import display_housing_prediction

# Import configuration
from config import MODEL_CONFIG

def load_housing_model():
    """Load the housing regression model from streamlit/models/housing_regression_model.pkl"""
    try:
        # Define the model path
        model_path = Path(__file__).parent.parent / 'models' / 'housing_regression_model.pkl'
        
        if not model_path.exists():
            st.error("No trained model found. Please train a model first.")
            return None
            
        try:
            # Load the model
            model = joblib.load(model_path)
            
            # Define expected features based on the training data
            expected_features = [
                'longitude', 'latitude', 'housing_median_age', 'total_rooms',
                'total_bedrooms', 'population', 'households', 'median_income',
                'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
                'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY',
                'ocean_proximity_NEAR OCEAN'
            ]
            
            # Create metadata
            metadata = {
                'features': expected_features,
                'target': 'median_house_value',
                'model_type': type(model).__name__
            }
            
            # Store model info in session state
            st.session_state.model_path = str(model_path)
            st.session_state.model_metadata = metadata
            
            # Display model info
            st.sidebar.success("✓ Model loaded successfully")
            st.sidebar.write(f"**Model type:** {type(model).__name__}")
            st.sidebar.write(f"**Target:** {metadata['target']}")
            
            return model
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def predict_housing_price(model, features):
    """Make a prediction using the loaded housing model."""
    try:
        # Create input DataFrame with correct feature order and names expected by the model
        input_data = {
            'MedInc': [features['median_income']],
            'HouseAge': [features['housing_median_age']],
            'AveRooms': [features['total_rooms'] / max(1, features['households'])],
            'AveBedrms': [features['total_bedrooms'] / max(1, features['households'])],
            'Population': [features['population']],
            'AveOccup': [features['population'] / max(1, features['households'])],
            'Latitude': [features['latitude']],
            'Longitude': [features['longitude']]
        }
        
        # Create DataFrame with correct feature order
        input_df = pd.DataFrame(input_data)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Calculate confidence interval (simplified example)
        confidence_interval = (prediction * 0.95, prediction * 1.05)
        
        return {
            'prediction': prediction,
            'confidence_interval': confidence_interval,
            'input_features': features  # Return original features for display
        }
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def main():
    """Main function for the housing price prediction page."""
    st.set_page_config(
        page_title="Housing Price Prediction",
        page_icon="🏠",
        layout="wide"
    )
    
    st.title("🏠 California Housing Price Prediction")
    st.write("Predict the median house value for a given set of features.")
    
    # Load model if not already loaded
    if 'model' not in st.session_state:
        st.session_state.model = load_housing_model()
    
    # Get user input features
    features = get_housing_features()
    
    # Make prediction when button is clicked
    if st.button("Predict House Value"):
        if 'model' in st.session_state and st.session_state.model is not None:
            with st.spinner("Making prediction..."):
                result = predict_housing_price(st.session_state.model, features)
                if result:
                    display_housing_prediction(result)
        else:
            st.error("Please load a model first.")
    
    # Add some space at the bottom
    st.write("")
    st.write("---")
    st.write("### About")
    st.write(
        "This model predicts the median house value for districts in California based on "
        "various features such as location, number of rooms, and median income."
    )

if __name__ == "__main__":
    main()
