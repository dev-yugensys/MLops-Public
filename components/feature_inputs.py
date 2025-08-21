"""
UI components for feature inputs in the YugenAI Streamlit app.
"""
import streamlit as st
from typing import Dict, Any

def number_input_with_slider(
    label: str,
    min_value: float,
    max_value: float,
    step: float,
    value: float,
    key: str,
    help_text: str = None
) -> float:
    """
    Create a number input with a slider for better UX.
    
    Args:
        label: Label for the input
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        step: Step size
        value: Default value
        key: Unique key for the input
        help_text: Optional help text
        
    Returns:
        The selected value
    """
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Slider for better UX
        value = st.slider(
            label=label,
            min_value=min_value,
            max_value=max_value,
            value=value,
            step=step,
            key=f"{key}_slider",
            help=help_text
        )
    
    with col2:
        # Number input for precise control
        value = st.number_input(
            label=" ",  # Empty label since we have it in the slider
            min_value=min_value,
            max_value=max_value,
            value=value,
            step=step,
            key=key,
            format="%.1f"
        )
    
    return value

def get_iris_features() -> Dict[str, float]:
    """
    Render input fields for Iris flower features.
    
    Returns:
        Dictionary containing the input features
    """
    st.subheader("Iris Flower Features")
    
    # Get feature configuration
    model_config = st.session_state.get('model_config', {})
    features_config = model_config.get('features', {})
    
    # Store features in session state if not already present
    if 'iris_features' not in st.session_state:
        st.session_state.iris_features = {
            'sepal_length': features_config.get('sepal_length', {}).get('value', 5.8),
            'sepal_width': features_config.get('sepal_width', {}).get('value', 3.0),
            'petal_length': features_config.get('petal_length', {}).get('value', 3.8),
            'petal_width': features_config.get('petal_width', {}).get('value', 1.2)
        }
    
    # Create input fields for each feature
    for feature_key, config in features_config.items():
        st.session_state.iris_features[feature_key] = number_input_with_slider(
            label=config['label'],
            min_value=config['min'],
            max_value=config['max'],
            step=config['step'],
            value=st.session_state.iris_features[feature_key],
            key=f"iris_{feature_key}",
            help_text=f"Enter the {config['label'].lower()}"
        )
    
    return st.session_state.iris_features

def get_housing_features() -> Dict[str, Any]:
    """
    Render input fields for housing features.
    
    Returns:
        Dictionary containing the input features
    """
    st.subheader("House Features")
    
    # Default values for housing features
    default_features = {
        'longitude': -122.25,
        'latitude': 37.85,
        'housing_median_age': 30.0,
        'total_rooms': 2000.0,
        'total_bedrooms': 400.0,
        'population': 1000.0,
        'households': 350.0,
        'median_income': 4.0,
        'ocean_proximity': 'NEAR BAY'
    }
    
    # Initialize session state for housing features if not exists
    if 'housing_features' not in st.session_state:
        st.session_state.housing_features = default_features.copy()
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Location features
        st.markdown("### Location")
        st.session_state.housing_features['longitude'] = number_input_with_slider(
            label="Longitude",
            min_value=-124.35,
            max_value=-114.31,
            step=0.1,
            value=st.session_state.housing_features['longitude'],
            key="longitude",
            help_text="Longitude of the house location"
        )
        
        st.session_state.housing_features['latitude'] = number_input_with_slider(
            label="Latitude",
            min_value=32.54,
            max_value=41.95,
            step=0.1,
            value=st.session_state.housing_features['latitude'],
            key="latitude",
            help_text="Latitude of the house location"
        )
        
        # House features
        st.markdown("### House Details")
        st.session_state.housing_features['housing_median_age'] = number_input_with_slider(
            label="Median Age of Houses",
            min_value=1.0,
            max_value=52.0,
            step=1.0,
            value=st.session_state.housing_features['housing_median_age'],
            key="housing_median_age",
            help_text="Median age of houses in the area"
        )
        
        st.session_state.housing_features['total_rooms'] = number_input_with_slider(
            label="Total Rooms",
            min_value=2.0,
            max_value=40000.0,
            step=100.0,
            value=st.session_state.housing_features['total_rooms'],
            key="total_rooms",
            help_text="Total number of rooms in the area"
        )
    
    with col2:
        # Room details
        st.markdown("### Room Details")
        st.session_state.housing_features['total_bedrooms'] = number_input_with_slider(
            label="Total Bedrooms",
            min_value=1.0,
            max_value=8000.0,
            step=50.0,
            value=st.session_state.housing_features['total_bedrooms'],
            key="total_bedrooms",
            help_text="Total number of bedrooms in the area"
        )
        
        st.session_state.housing_features['population'] = number_input_with_slider(
            label="Population",
            min_value=3.0,
            max_value=40000.0,
            step=100.0,
            value=st.session_state.housing_features['population'],
            key="population",
            help_text="Total population in the area"
        )
        
        st.session_state.housing_features['households'] = number_input_with_slider(
            label="Households",
            min_value=1.0,
            max_value=6000.0,
            step=50.0,
            value=st.session_state.housing_features['households'],
            key="households",
            help_text="Total number of households in the area"
        )
        
        st.session_state.housing_features['median_income'] = number_input_with_slider(
            label="Median Income (in tens of thousands)",
            min_value=0.5,
            max_value=15.0,
            step=0.1,
            value=st.session_state.housing_features['median_income'],
            key="median_income",
            help_text="Median income of households in the area"
        )
    
    # Ocean proximity (categorical)
    st.session_state.housing_features['ocean_proximity'] = st.selectbox(
        "Ocean Proximity",
        options=['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
        index=3,  # Default to NEAR BAY
        key="ocean_proximity",
        help="Proximity to the ocean"
    )
    
    return st.session_state.housing_features
