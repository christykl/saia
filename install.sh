# ------------------------------------------------------------------------------
# 1. Create and activate SAIA conda environment
# ------------------------------------------------------------------------------
conda env create -f environment.yml
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate saia

# ------------------------------------------------------------------------------
# 2. Install PyTorch (specific CUDA version) and related packages
#    Adjust CUDA version (cu121) if necessary
# ------------------------------------------------------------------------------
uv pip install -r pyproject.toml
mkdir -p saia/utils/FairFace/fair_face_models
gdown --id 113QMzQzkBDmYMs9LwzvD-jxEZdBQ5J4X -O saia/utils/FairFace/fair_face_models/fairface_alldata_20191111.pt

# ------------------------------------------------------------------------------
# 3. Install InstructDiffusion and download pretrained weights
# ------------------------------------------------------------------------------
git clone https://github.com/cientgu/InstructDiffusion.git ./saia/utils/InstructDiffusion
cd ./saia/utils/InstructDiffusion
bash scripts/download_pretrained_instructdiffusion.sh
cd ..  # Return to root project directory

# ------------------------------------------------------------------------------
# 4. Setup Grounded-Segment-Anything (GroundingDINO + SAM)
# ------------------------------------------------------------------------------
export AM_I_DOCKER="False"
export BUILD_WITH_CUDA="True"
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export CC="$(which gcc)"
export CXX="$(which g++)"

cd Grounded-Segment-Anything
uv pip install ./segment_anything
uv pip install -e ./GroundingDINO --no-build-isolation

# Download model weights
wget "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
wget "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
cd ..

# ------------------------------------------------------------------------------
# 5. Install Hugging Face libraries (latest versions)
# ------------------------------------------------------------------------------
uv pip install git+https://github.com/huggingface/transformers.git
uv pip install git+https://github.com/huggingface/diffusers.git