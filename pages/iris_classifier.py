"""
Iris Flower Classification Page
"""
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import os

# Import components and utilities
from components.feature_inputs import get_iris_features
from components.results_display import display_iris_prediction

# Import configuration
from config import MODEL_CONFIG, CLASS_NAMES

def load_iris_model():
    """Load the Iris classification model from streamlit/models/iris_classification_model.pkl"""
    try:
        # Define the model path
        model_path = Path(__file__).parent.parent / 'models' / 'iris_classification_model.pkl'
        
        if not model_path.exists():
            st.error("No trained model found. Please train a model first.")
            return None
            
        try:
            # Load the model directly - it's saved as a raw model, not in a dictionary
            model = joblib.load(model_path)
            
            # Create metadata with default values
            metadata = {
                'features': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                'classes': ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                'model_type': type(model).__name__
            }
            
            # If the model has feature names, use them
            if hasattr(model, 'feature_names_in_'):
                metadata['features'] = list(model.feature_names_in_)
                
            # If the model has class names, use them
            if hasattr(model, 'classes_'):
                metadata['classes'] = [str(c) for c in model.classes_]
            
            # Store model info in session state
            st.session_state.model_path = str(model_path)
            st.session_state.model_metadata = metadata
            
            # Display model info
            st.sidebar.success("✓ Model loaded successfully")
            st.sidebar.write(f"**Model type:** {type(model).__name__}")
            st.sidebar.write("**Features:** " + ", ".join(metadata['features']))
            st.sidebar.write("**Classes:** " + ", ".join(metadata['classes']))
            
            return model
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def predict_iris(model, features):
    """Make a prediction using the loaded model."""
    try:
        # Get metadata from session state
        metadata = getattr(st.session_state, 'model_metadata', {})
        expected_features = metadata.get('features', [])
        
        if not expected_features:
            st.sidebar.warning("⚠️ Model has no feature names, using default feature order")
            print("Using standard features as model has no feature names")
            expected_features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        
        # Create input DataFrame with correct feature order
        input_data = {}
        for feat in expected_features:
            # Map feature names to match input format
            if feat == 'SepalLengthCm':
                input_data[feat] = [features['sepal_length']]
            elif feat == 'SepalWidthCm':
                input_data[feat] = [features['sepal_width']]
            elif feat == 'PetalLengthCm':
                input_data[feat] = [features['petal_length']]
            elif feat == 'PetalWidthCm':
                input_data[feat] = [features['petal_width']]
            else:
                # For any additional features, use default value of 0
                input_data[feat] = [0.0]
        
        # Create DataFrame with correct feature order
        input_df = pd.DataFrame(input_data, columns=expected_features)
        
        # Make prediction
        print("\n=== Making Prediction ===")
        print(f"Input DataFrame shape: {input_df.shape}")
        print(f"Input DataFrame columns: {input_df.columns.tolist()}")
        
        # Print model information for debugging
        print("\n=== Model Information ===")
        print(f"Model type: {type(model).__name__}")
        
        if hasattr(model, 'predict_proba'):
            print("Calling predict_proba...")
            prediction_proba = model.predict_proba(input_df)[0]
            print(f"Raw prediction probabilities: {prediction_proba}")
            predicted_class_idx = np.argmax(prediction_proba)
            print(f"Predicted class index: {predicted_class_idx}")
            print(f"Predicted class probability: {np.max(prediction_proba):.4f}")
            
            # Also get prediction using predict for consistency check
            predicted_class_idx_predict = model.predict(input_df)[0]
            print(f"Predicted class index (from predict): {predicted_class_idx_predict}")
            
            # Ensure we're using consistent predictions
            if predicted_class_idx != predicted_class_idx_predict:
                print(f"Warning: predict and predict_proba disagree! Using predict_proba result: {predicted_class_idx}")
        else:
            # Fallback for models without probability estimates
            print("Model does not support predict_proba, using predict only")
            predicted_class_idx = model.predict(input_df)[0]
            print(f"Predicted class index: {predicted_class_idx}")
            prediction_proba = [0.0] * 3  # Default to 3 classes for Iris
            try:
                prediction_proba[predicted_class_idx] = 1.0
                print(f"Set probability for class {predicted_class_idx} to 1.0")
            except IndexError:
                error_msg = f"Predicted class index {predicted_class_idx} is out of range for classes [0, 1, 2]"
                print(error_msg)
                st.sidebar.warning(error_msg)
        
        # Define the iris species mapping
        iris_species = {
            '0': 'Iris-setosa',
            '1': 'Iris-versicolor',
            '2': 'Iris-virginica'
        }
        
        # Get class names from model or use defaults
        if hasattr(model, 'classes_'):
            class_indices = [str(c) for c in model.classes_]
            print(f"Model class indices: {class_indices}")
            
            # Map numeric indices to species names
            class_names = [iris_species.get(idx, f"Unknown-{idx}") for idx in class_indices]
            print(f"Mapped class names: {class_names}")
        else:
            class_names = list(iris_species.values())
            print(f"Using default class names: {class_names}")
        
        # Map predicted class index to class name
        try:
            predicted_class = class_names[predicted_class_idx]
            print(f"Mapped class index {predicted_class_idx} to species: {predicted_class}")
        except IndexError as e:
            error_msg = f"Class index {predicted_class_idx} out of range for classes {class_names}"
            print(error_msg)
            st.sidebar.error(error_msg)
            predicted_class = f"Unknown Class ({predicted_class_idx})"
        
        # Get the confidence score (probability of the predicted class)
        confidence_score = float(prediction_proba[predicted_class_idx]) * 100
        
        # Format the result
        result = {
            'prediction': {
                'class': predicted_class,
                'class_index': int(predicted_class_idx),
                'confidence': confidence_score,
                'probabilities': {
                    name: float(prob) 
                    for name, prob in zip(class_names, prediction_proba)
                },
                'features': {
                    'sepal_length': float(features.get('sepal_length', 0.0)),
                    'sepal_width': float(features.get('sepal_width', 0.0)),
                    'petal_length': float(features.get('petal_length', 0.0)),
                    'petal_width': float(features.get('petal_width', 0.0))
                },
                'model_info': {
                    'type': type(model).__name__,
                    'has_probabilities': hasattr(model, 'predict_proba'),
                    'feature_count': len(expected_features),
                    'input_features': input_data
                }
            }
        }
        
        # Update the UI to show the prediction and confidence
        st.success(f"Prediction: {predicted_class}")
        st.success(f"Confidence: {confidence_score:.2f}%")
        
        # Show the probabilities in an expander
        with st.expander("View detailed probabilities"):
            for name, prob in result['prediction']['probabilities'].items():
                st.write(f"- {name}: {prob*100:.2f}%")
        
        # Log prediction details to sidebar
        st.sidebar.write("### Prediction Details")
        st.sidebar.write(f"**Predicted Class:** {predicted_class}")
        st.sidebar.write("**Confidence:** {:.2f}%".format(confidence_score))
        st.sidebar.write("**Probabilities:**")
        for name, prob in result['prediction']['probabilities'].items():
            st.sidebar.write(f"- {name}: {prob*100:.2f}%")
        
        return result
        
    except Exception as e:
        error_msg = f"❌ Prediction error: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        st.sidebar.error(f"Prediction failed: {str(e)}")
        return None

def main():
    """Main function for the Iris classifier page."""
    # Set page title and description
    st.title("Iris Flower Classifier")
    st.markdown("""
        This application uses machine learning to classify iris flowers into three species:
        **Setosa**, **Versicolor**, and **Virginica** based on their sepal and petal measurements.
    """)
    
    # Initialize session state variables if they don't exist
    if 'class_names' not in st.session_state:
        st.session_state.class_names = CLASS_NAMES
    
    # Set up model configuration in session state
    if 'model_config' not in st.session_state:
        st.session_state.model_config = {
            'features': {
                'sepal_length': {'label': 'Sepal Length (cm)', 'min': 4.0, 'max': 8.0, 'step': 0.1, 'value': 5.8},
                'sepal_width': {'label': 'Sepal Width (cm)', 'min': 2.0, 'max': 4.5, 'step': 0.1, 'value': 3.0},
                'petal_length': {'label': 'Petal Length (cm)', 'min': 1.0, 'max': 7.0, 'step': 0.1, 'value': 3.8},
                'petal_width': {'label': 'Petal Width (cm)', 'min': 0.1, 'max': 2.5, 'step': 0.1, 'value': 1.2}
            }
        }
    
    # Debug: Show the loaded class names
    st.sidebar.write("=== Class Names ===")
    st.sidebar.write(CLASS_NAMES)
    
    # Load the model
    model = load_iris_model()
    
    # Get input features from the user
    st.subheader("Input Features")
    features = get_iris_features()
    
    # Add a button to make predictions
    if st.button("Classify Iris"):
        st.subheader("Prediction Results")
        
        # Show loading state and make prediction
        with st.spinner("Classifying iris flower..."):
            if model is not None:
                # Make prediction
                prediction = predict_iris(model, features)
                if prediction:
                    # Display prediction
                    display_iris_prediction(prediction)
                    
                    # Debug: Show the full prediction object
                    st.sidebar.write("=== Full Prediction ===")
                    st.sidebar.json(prediction)
            else:
                st.error("Failed to load the model. Please check the model file.")
    
    # Add a section for model information
    st.markdown("---")
    st.subheader("Model Information")
    st.info("This model classifies iris flowers based on their sepal and petal measurements.")
    st.write("**Model Type:** Scikit-learn Classifier")
    
    # Ensure class_names is properly set in session state
    if 'class_names' not in st.session_state:
        st.session_state.class_names = CLASS_NAMES
    
    # Display class information
    if isinstance(CLASS_NAMES, dict):
        st.write("**Classes:**", ", ".join(CLASS_NAMES.keys()))
    else:
        st.write("**Classes:**", ", ".join(CLASS_NAMES))

if __name__ == "__main__":
    main()
