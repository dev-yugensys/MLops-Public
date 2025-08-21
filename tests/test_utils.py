"""
Basic utility tests that don't require complex mocks
"""
import pytest
import pandas as pd
import numpy as np

def test_basic_imports():
    """Test that basic imports work"""
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    assert pd is not None
    assert np is not None
    assert Path is not None

def test_dataframe_creation():
    """Test basic DataFrame operations"""
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['a', 'b', 'c']
    })
    
    assert len(df) == 3
    assert list(df.columns) == ['A', 'B']
    assert df['A'].sum() == 6

# Add more basic utility tests here
