import pytest
from unittest.mock import MagicMock
from PIL import Image
import io

from src.main import create_app

# Create fixture for test client
@pytest.fixture
def client(mocker):
    mocker.patch('src.main.get_pipeline', return_value=MagicMock())
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# Fixture to create dummy image for mocking
@pytest.fixture
def dummy_image():
    """Creates a simple PIL Image for testing and returns it as a file-like object."""
    # Create the image
    img = Image.new('RGB', (100, 100), color='red')
    
    # Save the image to an in-memory binary buffer
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    
    # Reset the buffer's position to the beginning so it can be read
    buffer.seek(0)
    
    return buffer

# Fixture for dummy glb file for mocking
@pytest.fixture
def dummy_glb_file():
    """Creates a dummy in-memory binary .glb file."""
    dummy_content = b'glTF...dummy binary data for a 3d model...'
    
    # Create an in-memory binary buffer
    buffer = io.BytesIO(dummy_content)
    
    # Reset the buffer's position to the beginning, so it can be read
    buffer.seek(0)
    
    return buffer

# Fixture for failed 3D generation for mocking
@pytest.fixture
def failed_3d():
    """Returns None to simulate the 3D generation having failed."""
    return None

# Fixture with fully functional client for E2E test
@pytest.fixture
def full_client():
    """Fixture to provide a full client without any mocking."""
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client