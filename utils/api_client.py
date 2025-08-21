"""
API client for interacting with the YugenAI backend services.
"""
import requests
from typing import Dict, Any, Optional
import json
import logging

# Set up logging
logger = logging.getLogger(__name__)

class APIClient:
    """Client for making API requests to the YugenAI backend."""
    
    def __init__(self, base_url: str):
        """Initialize the API client with the base URL."""
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def predict_iris(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Make a prediction using the Iris classification model.
        
        Args:
            features: Dictionary containing iris flower features
            
        Returns:
            Dictionary containing prediction results
        """
        endpoint = f"{self.base_url}/predict_iris"
        payload = {
            "SepalLengthCm": features["sepal_length"],
            "SepalWidthCm": features["sepal_width"],
            "PetalLengthCm": features["petal_length"],
            "PetalWidthCm": features["petal_width"]
        }
        
        try:
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {"error": str(e)}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get the health status of the API."""
        endpoint = f"{self.base_url}/health"
        
        try:
            response = self.session.get(endpoint)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting health status: {str(e)}")
            return {"status": "unhealthy", "error": str(e)}
    
    def get_model_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics."""
        endpoint = f"{self.base_url}/metrics"
        
        try:
            response = self.session.get(endpoint)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting model metrics: {str(e)}")
            return {"error": str(e)}
