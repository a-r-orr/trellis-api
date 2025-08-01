import os
# import io

# Set environment variables before importing torch and other libraries
os.environ['SPCONV_ALGO'] = 'auto'
os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0' 

# from werkzeug.datastructures import FileStorage
# from PIL import Image
import torch

# Import TRELLIS-specific libraries
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

# Global variable to hold the pipeline
# pipeline = None

def get_pipeline():
    """Loads and returns the TRELLIS pipeline, loading only if it hasn't been loaded yet."""
    # global pipeline
    
    # if pipeline is None:
    print("Loading TRELLIS model for the first time...")
    torch.cuda.empty_cache()
    # Load the pipeline from Hugging Face
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    # Move the pipeline to the GPU
    # pipeline.cuda()
    print("Model loaded successfully.")

    return pipeline


def create_3d_model(image, gen_pipeline):
    """
    Generates a 3D model from the provided image using the TRELLIS pipeline.
    """
    # Ensure cache is clear before running the model
    torch.cuda.empty_cache()

    try:
        # Run the pipeline to get 3D assets
        outputs = gen_pipeline.run(
            image,
            seed=1,
            # Optional parameters
            sparse_structure_sampler_params={
                "steps": 30,
                "cfg_strength": 7.5,
            },
            slat_sampler_params={
                "steps": 30,
                "cfg_strength": 3,
            },
        )

        # Convert the output to a GLB file format in memory
        print("Creating GLB file...")
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=0.95,      # Ratio of triangles to remove
            texture_size=1024,  # Size of the texture for the GLB
        )
        torch.cuda.empty_cache()
        return glb
    except Exception as e:
        print(f"3D Generation failed: {e}")
        torch.cuda.empty_cache()
        return None