import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock Streamlit and other external dependencies
@pytest.fixture(autouse=True)
def mock_imports():
    with patch.dict('sys.modules', {
        'streamlit': MagicMock(),
        'streamlit.runtime': MagicMock(),
        'streamlit.runtime.scriptrunner': MagicMock(),
        'preprocessing': MagicMock(),
    }):
        yield
