import pytest
from PIL import Image
from unittest.mock import MagicMock
import io
import pathlib

def test_create_from_image_endpoint(client, mocker, dummy_glb_file, dummy_image):

    # Mock the main.create_image function to enable faster testing
    mock_create = mocker.patch('src.api.create_3d_model', return_value=dummy_glb_file)
    dummy_glb_file.export = MagicMock()
    def export_side_effect(buffer, file_type):
        buffer.write(dummy_glb_file.read())

    # Configure the mock's .export method to use this side effect
    dummy_glb_file.export.side_effect = export_side_effect

    data = {
        'image_file': (dummy_image, 'test_image.png')
    }
    # Simulate request to the endpoint
    response = client.post('/models/create-from-image', data=data, content_type='multipart/form-data')

    # Check the response
    assert response.status_code == 200
    assert response.mimetype == 'model/gltf-binary'

    # Check that the mock was called correctly
    mock_create.assert_called_once()
    mock_create.assert_called_with(mocker.ANY, mocker.ANY)

    dummy_glb_file.export.assert_called_once()

    dummy_glb_file.seek(0)
    assert response.data == dummy_glb_file.read()


def test_failure_create_from_image_endpoint(client, mocker, failed_3d, dummy_image):
    
    # Mock the main.create_image function to enable faster testing
    mock_create = mocker.patch('src.api.create_3d_model', return_value=failed_3d)

    data = {
        'image_file': (dummy_image, 'test_image.png')
    }
    # Simulate request to the endpoint
    response = client.post('/models/create-from-image', data=data, content_type='multipart/form-data')

    # Check the response
    assert response.status_code == 500
    assert response.json == {"message": "3D generation failed on the server."}


def test_missing_image(client):
    """Test that an empty request data returns a 400 Bad Request"""
    response = client.post('/models/create-from-image', data='', content_type='multipart/form-data')

    assert response.status_code == 400
    assert response.json.get('message') == "Input payload validation failed"


def test_invalid_image(client):
    """Test that an invalid image returns a 400 Bad Request"""
    invalid_file_data = io.BytesIO(b"Text, not an image")
    data = {
        'image_file': (invalid_file_data, 'test.txt')
    }
    # Simulate request to the endpoint
    response = client.post('/models/create-from-image', data=data, content_type='multipart/form-data')

    assert response.status_code == 400
    assert 'Failed to read image file' in response.json.get('message')


# Final E2E test to check the ML functionality and confirm that a 3D .glb file is returned.
@pytest.mark.e2e
def test_e2e_3d_generation(full_client):
    """
    End-to-end test - tests the actual operation of the 3D generation.
    This test is slow and needs a GPU.
    """

    current_dir = pathlib.Path(__file__).parent
    image_path = current_dir / "test_image.png"

    with open(image_path, "rb") as image_file:
        data = {
            'image_file': (image_file, 'test_image.png')
        }
        response = full_client.post(
            '/models/create-from-image',
            data=data,
            content_type='multipart/form-data'
        )

    # Check the response
    assert response.status_code == 200
    assert response.mimetype == 'model/gltf-binary'
    assert len(response.data) > 0