#!/usr/bin/env bash
set -eo pipefail

# Install Cython first, as it's a build dependency for other packages
pip install --no-cache-dir "Cython>=0.29.37"

# -- Start of 'basic' install --
pip install --no-cache-dir gunicorn Flask flask-cors flask-restx uvicorn Werkzeug Jinja2 \
diffusers huggingface-hub safetensors tokenizers scikit-learn \
fastapi fastjsonschema numpy onnxruntime prompt_toolkit pygltflib PyMatting pyparsing \
imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime-gpu trimesh open3d xatlas pyvista pymeshfix igraph transformers

pip install --no-cache-dir "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# -- Start of 'train' install --
pip install --no-cache-dir tensorboard pandas lpips

# -- Start of other dependencies --
# Add --no-build-isolation to fix the "torch not found" error
pip install --no-build-isolation --no-cache-dir flash_attn xformers
# Set the environment variable to bypass the version check for Kaolin
IGNORE_TORCH_VER=1 pip install --no-cache-dir "kaolin @ git+https://github.com/NVIDIAGameWorks/kaolin.git@82e5f37be076bb9254aef519b6c9b8269f91620b"
pip install --no-cache-dir spconv-cu120

# Create directory for git clones
mkdir -p /tmp/extensions

# -- Install nvdiffrast --
git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
pip install --no-cache-dir /tmp/extensions/nvdiffrast

# -- Install diffoctreerast --
git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast
pip install --no-cache-dir /tmp/extensions/diffoctreerast

# -- Install mipgaussian --
git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
pip install --no-cache-dir /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/

# -- Install vox2seq --
pip install --no-cache-dir /tmp/extensions/vox2seq

# -- Clean up --
rm -rf /tmp/extensions