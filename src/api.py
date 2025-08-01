import os
import io

# Set environment variables before importing torch and other libraries
os.environ['SPCONV_ALGO'] = 'auto'
os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0' 

from flask import send_file, current_app
from flask_restx import Resource, Namespace
from werkzeug.datastructures import FileStorage
from PIL import Image
import torch

from .ml_logic import create_3d_model

# Create 3D Models Namespace
ns_3d = Namespace('models', description='3D Model operations')

# Parser for file upload
file_upload_parser = ns_3d.parser()
file_upload_parser.add_argument('image_file', 
                                location='files', 
                                type=FileStorage, 
                                required=True, 
                                help='PNG image for 3D model generation.')

@ns_3d.route('/create-from-image')
@ns_3d.expect(file_upload_parser)
class Gen3D(Resource):
    @ns_3d.doc('create_3d_from_image')
    @ns_3d.produces(['model/gltf-binary'])
    def post(self):
        '''Route for creating a new 3D model based on an image'''
        args = file_upload_parser.parse_args()
        uploaded_file = args['image_file']
        
        # Ensure it's a valid file
        if not uploaded_file:
            return {'message': 'No file provided'}, 400
            
        # Load the uploaded image using PIL
        try:
            input_image = Image.open(uploaded_file.stream)
        except Exception as e:
            return {'message': f'Failed to read image file: {e}'}, 400
        
        # Move the pipeline instance to the gpu before processing.
        current_app.model_pipeline.to("cuda")
        try:
            # Generate the 3D model
            glb_data = create_3d_model(input_image, current_app.model_pipeline)
        except Exception as e:
            print(e)
            return {'message': f'Failed to generate 3d file: {e}'}, 500
        
        if glb_data is None:
            return {'message': '3D generation failed on the server.'}, 500

        # Return the pipeline instance to the cpu after processing to free up VRAM.
        current_app.model_pipeline.to("cpu")

        # Prepare the GLB file for sending in the response
        buffer = io.BytesIO()
        # The glb object from postprocessing_utils can export directly to a file-like object
        glb_data.export(buffer, file_type='glb')
        buffer.seek(0)

        torch.cuda.empty_cache()

        return send_file(
            buffer, 
            mimetype='model/gltf-binary',
            as_attachment=True,
            download_name='generated_model.glb'
        )