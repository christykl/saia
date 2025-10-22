conda_base() {
  # 1) If the conda command exists, trust it
  if command -v conda >/dev/null 2>&1; then
    conda info --base 2>/dev/null && return 0
  fi

  # 2) If inside a conda env, derive base from CONDA_PREFIX
  if [ -n "$CONDA_PREFIX" ]; then
    base="$CONDA_PREFIX"
    # If we're in a non-base env, CONDA_PREFIX looks like <base>/envs/<name>
    case "$base" in
      */envs/*) base="${base%/envs/*}";;
    esac
    if [ -f "$base/etc/profile.d/conda.sh" ] || [ -d "$base/condabin" ] || [ -d "$base/bin" ]; then
      printf '%s\n' "$base"
      return 0
    fi
  fi

  # 3) Check common install locations
  local candidates=(
    "$HOME/miniconda3"
    "$HOME/anaconda3"
    "$HOME/miniforge3"
    "/opt/conda"
  )
  for c in "${candidates[@]}"; do
    if [ -f "$c/etc/profile.d/conda.sh" ] || [ -d "$c/condabin" ]; then
      printf '%s\n' "$c"
      return 0
    fi
  done

  # 4) Search for any folder that contains etc/profile.d/conda.sh
  # (keep search bounded for speed; widen or add roots if needed)
  local roots=("$HOME" "/opt" "/usr/local" "/usr")
  # shellcheck disable=SC2048,SC2086
  while IFS= read -r conda_sh; do
    # Strip the trailing /etc/profile.d/conda.sh to get the base
    printf '%s\n' "${conda_sh%/etc/profile.d/conda.sh}"
    return 0
  done < <(find ${roots[*]} -maxdepth 5 -path "*/etc/profile.d/conda.sh" -type f 2>/dev/null)

  # Not found
  return 1
}

# ------------------------------------------------------------------------------
# 1. Create and activate SAIA conda environment
# ------------------------------------------------------------------------------
source "$(conda_base)/etc/profile.d/conda.sh"
conda env create -f environment.yml
conda activate saia

# ------------------------------------------------------------------------------
# 2. Install PyTorch (specific CUDA version) and related packages
#    Adjust CUDA version (cu121) if necessary
# ------------------------------------------------------------------------------
uv pip install -r pyproject.toml
mkdir -p utils/FairFace/fair_face_models
gdown --id 113QMzQzkBDmYMs9LwzvD-jxEZdBQ5J4X -O utils/FairFace/fair_face_models/fairface_alldata_20191111.pt

# ------------------------------------------------------------------------------
# 3. Install InstructDiffusion and download pretrained weights
# ------------------------------------------------------------------------------
git clone https://github.com/cientgu/InstructDiffusion.git utils/InstructDiffusion
cd utils/InstructDiffusion
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