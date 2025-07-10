# Automated Detection of Visual Attribute Reliance with a Self-Reflective Agent #
## ICML 2025 Workshop on Actionable Interpretability ##

[Christy Li](https://christykl.github.io/), [Josep Lopez Camuñas](https://yusepp.github.io/), [Jake Touchet](https://www.linkedin.com/in/jake-touchet-557329297/), [Jacob Andreas](https://www.mit.edu/~jda/), [Agata Lapedriza Garcia](https://s3.sunai.uoc.edu/web/agata/index.html), [Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/), [Tamar Rott Shaham](https://tamarott.github.io/)

<img src="assets/teaser.jpg" alt="figure 1">

We use a Self-reflective Automated Interpretability Agent (SAIA) to produce natural-language descriptions of the visual attributes that a model relies on to recognize or detect a given concept. For each target concept (e.g., *vase*, *teacher*, or *pedestrian*), the agent first conducts hypothesis testing to reach a candidate description and then validates the description's predictiveness of actual model behavior through a self-evaluation protocol. The top row shows the agent’s generated explanations. The bottom rows show images predicted to elicit high (green) or low (red) scores, along with their actual model confidence scores. Results are shown for different target concepts across an object recognition model with a controlled attribute reliance (left), CLIP (middle), and YOLOv8 (right).

## Installations ##
### General Dependencies ###
Clone this repo and create a conda environment:
```bash
git clone https://github.com/christykl/saia.git
cd saia
conda create -n saia_env python=3.10 --file conda_packages.txt -c nvidia
conda activate saia_env
```

Install packages and dependencies
```bash
pip install -r torch_requirements.txt
pip install -r requirements.txt
pip install -r torch_requirements.txt --force-reinstall
pip install git+https://github.com/huggingface/transformers.git
```

Install InstructDiffusion and Flux
```bash
cd utils
git clone https://github.com/cientgu/InstructDiffusion.git
pip install -r requirements_instdiff_flux.txt
cd InstructDiffusion
bash scripts/download_pretrained_instructdiffusion.sh
cd ../../
```
### Set Up Attribute Reliance Detection Benchmark Models ###
Set environment variables for SAM and Grounding DINO
```bash
export AM_I_DOCKER="False"
export BUILD_WITH_CUDA="True"
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export CC=$(which gcc-12)
export CXX=$(which g++-12)
```

Install SAM and Grounding DINO
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

Download model checkpoints
```bash
cd utils/Grounded-Segment-Anything

# Segment Anything (SAM)
wget "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# Grounding DINO
wget "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

cd ../../
```

Download precomputed exemplars for the attribute reliance benchmark models:
```bash
bash download_exemplars.sh
```

If you want to use the demographic attribute reliant models, download the FairFace models from [here](https://drive.google.com/drive/folders/1F_pXfbzWvG-bhCpNsRj6F_xsdjpesiFu?usp=sharing) into the ```./FairFace/``` folder.

## Quick Start ##
You can run demo experiments on individual units using ```demo.ipynb```:
\
\
Launch Jupyter Notebook
```bash
jupyter notebook
```
This command will start the Jupyter Notebook server and open the Jupyter Notebook interface in your default web browser. The interface will show all the notebooks, files, and subdirectories in this repo (assuming is was initiated from the root of the repo directory). Open ```demo.ipynb``` and proceed according to the instructions.

## Batch experimentation ##
To run a batch of experiments, use ```main.py```:

### Load OpenAI or Anthropic API key ###
(you can get an OpenAI API key by following the instructions [here](https://platform.openai.com/docs/quickstart) and an Anthropic API key by following the instructions [here](https://docs.anthropic.com/en/docs/get-started)).

Set your API key as an environment variable
```bash
export OPENAI_API_KEY='your-openai-api-key-here'
export ANTHROPIC_API_KEY='your-anthropic-api-key-here'
```

### Load Huggingface key ###
You will need a Huggingface API key if you want to use Stable Diffusion 3.5 as the text2image model (you can get a HuggingFace API key by following the instructions [here](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)).

Set your API key as an environment variable
```bash
export HF_TOKEN='your-hf-token-here'
```

### Run Agent ###
To run the model on a benchmark model, specify the ```bias_mode```, ```bias```, and ```classifiers``` by calling e.g.:
```bash
python main.py --bias_mode setting --bias beach --classifiers 0 1
``` 
Refer to the ```./exemplars/``` folder to choose the classifier numbers based on desired target concepts, e.g. classifier 0 of the beach setting-dependent system refers to the target concept "bench", and classifier 1 of the same system refers to the target concept "bird".

To run the model on CLIP, set ```bias_mode``` and ```bias``` to ```clip```, and ```classifiers``` by calling e.g.:
```bash
python main.py --bias_mode setting --bias beach --classifiers "scientist" "artist"
``` 
Refer to the documentation of ```main.py``` for more configuration options.

Results are automatically saved to an html file under ```./results/``` and can be viewed in your browser by starting a local server:
```bash
python -m http.server 80
```
Once the server is up, open the html in [http://localhost:80](http://localhost:80/results/)