import pytest
from unittest.mock import MagicMock
from src.ml_logic import create_3d_model

def test_create_3d_model_pipeline_logic(mocker):
    """
    Unit tests the create_3d_model function to ensure the pipeline is called with the correct paramaters
    """
    # Create mock image and model
    mock_image = MagicMock()
    mock_pipeline = MagicMock()
    mock_glb_function = mocker.patch('src.ml_logic.postprocessing_utils.to_glb', return_value="glb_output")

    # Set return value for mock pipeline
    mock_pipeline.run.return_value = {
        'gaussian': ["gaussian output"],
        'mesh': ["mesh output"]
    }

    # Call the function with the mocks
    result = create_3d_model(mock_image, mock_pipeline)
    assert result == "glb_output"

    mock_pipeline.run.assert_called_once_with(
        mock_image,
        seed=1,
        # Optional parameters
        sparse_structure_sampler_params={
            "steps": 25,
            "cfg_strength": 7.5,
        },
        slat_sampler_params={
            "steps": 15,
            "cfg_strength": 3,
        },
    )
    mock_glb_function.assert_called_once_with(
        "gaussian output",
        "mesh output",
        simplify=0.95,
        texture_size=1024,
    )