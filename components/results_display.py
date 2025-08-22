"""
UI components for displaying prediction results.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

def display_iris_prediction(prediction: Dict[str, Any]):
    """
    Display the Iris flower classification results.
    
    Args:
        prediction: Dictionary containing prediction results
    """
    if not prediction or "error" in prediction:
        st.error("Error making prediction. Please try again.")
        if "error" in prediction:
            st.error(f"Error details: {prediction['error']}")
        return
    
    # Get class names and colors from config
    from config import CLASS_NAMES
    
    # Get the prediction data
    predicted_class = str(prediction.get("class", "Unknown")).strip()
    probabilities = prediction.get("probabilities", {})
    
    # Debug print
    print(f"\n=== Displaying Prediction ===")
    print(f"Raw prediction: {prediction}")
    print(f"Predicted class: {predicted_class}")
    print(f"Probabilities: {probabilities}")
    
    # Use the class names from config
    class_names = CLASS_NAMES.copy()
    
    # If we have numeric probabilities but no class names, create them
    if probabilities and not any(isinstance(k, str) for k in probabilities.keys()):
        print("Numeric probabilities detected, mapping to class names")
        mapped_probs = {}
        for i, prob in enumerate(probabilities):
            class_name = f"Iris-{i}"  # Default name if not found
            # Try to find a matching class name
            for name in class_names.keys():
                if str(i) in name or name.endswith(str(i)):
                    class_name = name
                    break
            mapped_probs[class_name] = float(prob)
        probabilities = mapped_probs
        print(f"Mapped probabilities: {probabilities}")
    
    # Ensure all predicted classes are in class_names
    for class_name in list(probabilities.keys()):
        if class_name not in class_names:
            display_name = class_name.replace('Iris-', '').title()
            class_names[class_name] = {
                "display": display_name,
                "color": "#666666"
            }
    
    # If it's a numeric index, try to map it to a class name
    if predicted_class.isdigit():
        idx = int(predicted_class)
        if idx < len(class_names):
            predicted_class = list(class_names.keys())[idx]
        else:
            # Try to find a class name that matches this index
            for name in class_names.keys():
                if str(idx) in name or name.endswith(str(idx)):
                    predicted_class = name
                    break
    
    # Calculate confidence
    if not probabilities and 'confidence' in prediction:
        confidence = float(prediction['confidence'])
    else:
        confidence = max(probabilities.values(), default=0) * 100 if probabilities else 0
    
    # Get display name and color for the predicted class
    class_info = class_names.get(predicted_class, {
        "display": predicted_class, 
        "color": "#666666"
    })
    
    # Handle case where class_info is not a dictionary
    if not isinstance(class_info, dict):
        class_info = {"display": str(class_info), "color": "#666666"}
        
    display_name = class_info.get("display", predicted_class)
    color = class_info.get("color", "#666666")
    
    # Display prediction with colored box
    st.markdown(
        f"""
        <div class="prediction-box" style="border-left: 5px solid {color};">
            <h3>Prediction: <span style="color: {color};">{display_name}</span></h3>
            <p>Confidence: <strong>{confidence:.1f}%</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display class probabilities as a bar chart
    class_probs = prediction.get("probabilities", {})
    if class_probs:
        # Convert to list of dicts for easier plotting
        prob_data = []
        for class_name, prob in class_probs.items():
            display_name = class_names.get(class_name, {"display": class_name})["display"]
            prob_data.append({
                "Class": display_name,
                "Probability": prob * 100,
                "color": class_names.get(class_name, {"color": "#666666"})["color"]
            })
        
        # Create bar chart
        fig = px.bar(
            prob_data,
            x="Class",
            y="Probability",
            color="Class",
            color_discrete_map={d["Class"]: d["color"] for d in prob_data},
            title="Class Probabilities",
            labels={"Probability": "Probability (%)", "Class": "Iris Species"},
            range_y=[0, 100]
        )
        
        # Customize the chart
        fig.update_layout(
            showlegend=False,
            xaxis_title=None,
            yaxis_title="Probability (%)",
            yaxis=dict(tickformat=".0f"),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)

def display_model_metrics(metrics: Dict[str, Any]):
    """
    Display model performance metrics.
    
    Args:
        metrics: Dictionary containing model metrics
    """
    if not metrics or "error" in metrics:
        st.warning("Unable to load model metrics.")
        return
    
    st.subheader("Model Performance")
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.1f}%")
    
    with col2:
        st.metric("Precision", f"{metrics.get('precision', 0) * 100:.1f}%")
    
    with col3:
        st.metric("Recall", f"{metrics.get('recall', 0) * 100:.1f}%")
    
    # Add more detailed metrics in an expander
    with st.expander("View Detailed Metrics"):
        st.json(metrics)

def display_housing_prediction(prediction: Dict[str, Any]):
    """
    Display the housing price prediction results.
    
    Args:
        prediction: Dictionary containing prediction results
    """
    if not prediction or "error" in prediction:
        st.error("Error making prediction. Please try again.")
        if "error" in prediction:
            st.error(f"Error details: {prediction['error']}")
        return
    
    # Get prediction details
    predicted_value = prediction.get("prediction", 0)
    confidence_interval = prediction.get("confidence_interval", (0, 0))
    input_features = prediction.get("input_features", {})
    
    # Format the predicted value as currency
    formatted_value = "${:,.2f}".format(predicted_value)
    
    # Display prediction with a nice card
    st.markdown(
        f"""
        <div style="
            border-radius: 10px;
            padding: 20px;
            background-color: #1f77b4;
            color: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        ">
            <h3 style="margin-top: 0; color: white;">Predicted House Value</h3>
            <p style="font-size: 28px; font-weight: bold; margin-bottom: 15px; color: white;">{formatted_value}</p>
            <p style="color: rgba(255,255,255,0.9); margin-bottom: 0; font-size: 16px;">
                Confidence: ${int(confidence_interval[0]):,} - ${int(confidence_interval[1]):,}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display input features in an expander
    with st.expander("View Input Features", expanded=False):
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Location")
            st.metric("Longitude", f"{input_features.get('longitude', 0):.4f}°")
            st.metric("Latitude", f"{input_features.get('latitude', 0):.4f}°")
            st.metric("Ocean Proximity", input_features.get('ocean_proximity', 'N/A'))
        
        with col2:
            st.markdown("### House Details")
            st.metric("Median Age", f"{input_features.get('housing_median_age', 0)} years")
            st.metric("Total Rooms", int(input_features.get('total_rooms', 0)))
            st.metric("Total Bedrooms", int(input_features.get('total_bedrooms', 0)))
            st.metric("Households", int(input_features.get('households', 0)))
            st.metric("Population", int(input_features.get('population', 0)))
            st.metric("Median Income", f"${input_features.get('median_income', 0):.2f}K")
    
    # Add some space at the bottom
    st.markdown("""
        <style>
            .stExpander {
                margin-bottom: 20px;
            }
        </style>
    """, unsafe_allow_html=True)
