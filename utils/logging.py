"""Utility functions for logging model predictions and requests."""
import json
from datetime import datetime
from typing import Dict, Any, Optional
from .db import log_request, update_request

def log_model_prediction(user_id: str, model_name: str, 
                       input_data: Dict[str, Any], 
                       output_data: Dict[str, Any] = None,
                       error: Optional[Exception] = None) -> int:
    """
    Log a model prediction to the database.
    
    Args:
        user_id: ID of the user making the request
        model_name: Name of the model being used
        input_data: Input data for the model
        output_data: Output from the model (if successful)
        error: Exception if the prediction failed
        
    Returns:
        int: The ID of the logged request
    """
    try:
        # Log the initial request
        request_id = log_request(
            user_id=user_id,
            model_name=model_name,
            input_data=input_data
        )
        
        # If there was an error, update the request with error details
        if error is not None:
            update_request(
                request_id=request_id,
                status='failed',
                error=str(error)
            )
        # If we have output data, update the request with the results
        elif output_data is not None:
            update_request(
                request_id=request_id,
                output_data=output_data,
                status='completed'
            )
            
        return request_id
        
    except Exception as e:
        # Log any errors that occur during logging
        print(f"Error logging model prediction: {str(e)}")
        return -1
