import warnings
from flask import Flask
from flask_restx import Api
from flask_cors import CORS

from .ml_logic import get_pipeline
from .api import ns_3d

warnings.filterwarnings("ignore", message="TORCH_CUDA_ARCH_LIST is not set.*")
warnings.filterwarnings("ignore", message="xFormers is not available*")

def create_app():
    """Creates and configures the Flask app"""
    app = Flask(__name__)
    CORS(app)

    # Increase the maximum content length for file uploads to 16 MB (Can be increased if required)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

    # Initialise the API
    api = Api(app, version='1.0', title='Image to 3D API', 
              description='Returns a 3D Model (.glb) generated from a provided PNG image.')

    # Add the 3D Namespace
    api.add_namespace(ns_3d)

    # Load Pipeline and attach to the app instance
    with app.app_context():
        app.model_pipeline = get_pipeline()
    
    return app

if __name__ == "__main__":
     app = create_app()
     app.run(host='0.0.0.0', port=5050, debug=False)

# Command to run locally: python3 -m gunicorn --bind 127.0.0.1:5050 -w 1 --timeout 300 "src.main:create_app()"