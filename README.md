# Automated Detection of Visual Attribute Reliance with a Self-Reflective Agent #
### NeurIPS 2025 ###

### [Project Page](https://christykl.github.io/saia-website/)

[Christy Li](https://christykl.github.io/), [Josep Lopez Camuñas](https://yusepp.github.io/), [Jake Touchet](https://www.linkedin.com/in/jake-touchet-557329297/), [Jacob Andreas](https://www.mit.edu/~jda/), [Agata Lapedriza Garcia](https://s3.sunai.uoc.edu/web/agata/index.html), [Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/), [Tamar Rott Shaham](https://tamarott.github.io/)

<img src="assets/teaser.jpg" alt="figure 1">

We introduce the Self-reflective Automated Interpretability Agent (SAIA), a fully automated framework designed to detect visual attribute reliance in pretrained vision models. Given a pretrained model and a target visual concept (e.g., an image classifier selective for the object `vase`), SAIA identifies specific image features that systematically influence the model’s predictions, even when these features fall outside the model’s intended behavior (e.g., *the classifier relies on flowers to detect the vase*). At the core of our approach is a LM-based self-reflective agent that treats the task as a scientific discovery process. Rather than relying on a predefined set of candidate attributes, SAIA autonomously formulates hypotheses about image features that the model might rely on, designs targeted tests, and updates its beliefs based on observed model behavior. After generating an initial finding, SAIA actively evaluates how well it matches the model’s behavior on unseen test cases. If discrepancies arise during the self-evaluation, SAIA reflects on its assumptions, identifies gaps or inconsistencies in its current understanding, and initiates a new hypothesis testing loop.

## Installations ##
After cloning this repo, simply run
```bash
bash install.sh
```

To download precomputed exemplars for the attribute reliance benchmark models, run
```bash
bash download_exemplars.sh
```

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

To run the model on CLIP, set ```task```, ```bias_mode```, and ```bias``` to ```clip```, and use ```labels``` to specify the target concepts by calling e.g.:
```bash
python main.py --task clip --bias_mode clip --bias clip --labels "scientist" "artist"
``` 
Refer to the documentation of ```main.py``` for more configuration options.

Results are automatically saved to an html file under ```./results/``` and can be viewed in your browser by starting a local server:
```bash
python -m http.server 80
```
Once the server is up, open the html in [http://localhost:80](http://localhost:80/results/)
