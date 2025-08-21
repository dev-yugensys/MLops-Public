"""
Basic tests for the data preparation module
"""
import pytest
import pandas as pd
import numpy as np

def test_imports():
    """Test that required packages can be imported"""
    assert pd is not None
    assert np is not None

def test_dataframe_creation():
    """Test basic DataFrame operations"""
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['a', 'b', 'c']
    })
    assert len(df) == 3
    assert list(df.columns) == ['A', 'B']

def test_numpy_operations():
    """Test basic NumPy operations"""
    arr = np.array([1, 2, 3])
    assert arr.sum() == 6
