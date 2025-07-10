# Standard library imports
import math
import os
import time
import sys
from io import BytesIO
from typing import Dict, List, Tuple, Union

# Third-party imports
import openai
import requests
import torch
import torch.nn.functional as F
from baukit import Trace
from diffusers import (
    AutoPipelineForText2Image,
    EulerAncestralDiscreteScheduler,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionPipeline,
    StableDiffusion3Pipeline,
)
from PIL import Image

# Local imports
from utils.call_agent import ask_agent
from utils.api_utils import is_base64, format_api_content, image2str, str2image
from utils.SyntheticExemplars import SyntheticExemplars

# New imports
from utils.instdiff import InstructDiffusion
from utils.flux import FluxImageGenerator

sys.path.append('./utils/Grounded-Segment-Anything/')
from utils.SyntheticClassifiers import SAMClassifier
from utils.CLIPClassifiers import CLIPClassifier


class Synthetic_System:
    """
    A Python class containing the vision model and the specific classifier to interact with.
    
    Attributes
    ----------
    classifier_num : int
        The serial number of the classifier.
    layer : string
        The name of the layer where the classifier is located.
    model_name : string
        The name of the vision model.
    model : nn.Module
        The loaded PyTorch model.
    classifier : callable
        A lambda function to compute classifier activation and activation map per input image. 
        Use this function to test the classifier activation for a specific image.
    device : torch.device
        The device (CPU/GPU) used for computations.

    Methods
    -------
    load_model(model_name: str)->nn.Module
        Gets the model name and returns the vision model from PyTorch library.
    call_classifier(image_list: List[torch.Tensor])->Tuple[List[int], List[str]]
        returns the classifier activation for each image in the input image_list as well as the activation map 
        of the classifier over that image, that highlights the regions of the image where the activations 
        are higher (encoded into a Base64 string).
    """
    def __init__(self, classifier_num: int, classifier_labels: str, bias_mode: str, device: str, bias=None, bias_discount=None):
        
        self.classifier_num = classifier_num
        self.classifier_labels = classifier_labels
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")      
        self.threshold = 0
        self.layer = bias_mode
        if bias_mode == 'clip':
            self.classifier = CLIPClassifier(classifier_labels, bias_mode, bias='clip', bias_discount=0, device=self.device) 
        else:
            self.classifier = SAMClassifier(classifier_labels, bias_mode, bias=bias, bias_discount=bias_discount, device=self.device) 


    def call_classifier(self, image_list: List[str])->Tuple[List[float], List[str]]:
        """
        The function returns the classifier’s maximum activation value (in int format) over each of the images in the list as well as the activation map of the classifier over each of the images that highlights the regions of the image where the activations are higher (encoded into a Base64 string).
        
        Parameters
        ----------
        image_list : List[str]
            The input images in Base64 encoded string format
        
        Returns
        -------
        Tuple[List[float], List[str]]
            For each image in image_list returns the maximum activation value of the classifier on that image, and a masked images, 
            with the region of the image that caused the high activation values highlighted (and the rest of the image is darkened). Each image is encoded into a Base64 string.

        
        Examples
        --------
        >>> # test the activation value of the classifier for the prompt "a dog standing on the grass"
        >>> def execute_command(system, prompt_list) -> Tuple[int, str]:
        >>>     prompt = ["a dog standing on the grass"]
        >>>     image = text2image(prompt)
        >>>     activation_list, activation_map_list = system.call_classifier(image)
        >>>     return activation_list, activation_map_list
        >>> # test the activation value of the classifier for the prompt “a fox and a rabbit watch a movie under a starry night sky” “a fox and a bear watch a movie under a starry night sky” “a fox and a rabbit watch a movie at sunrise”
        >>> def execute_command(system.classifier, prompt_list) -> Tuple[int, str]:
        >>>     prompt_list = [[“a fox and a rabbit watch a movie under a starry night sky”, “a fox and a bear watch a movie under a starry night sky”,“a fox and a rabbit watch a movie at sunrise”]]
        >>>     images = text2image(prompt_list)
        >>>     activation_list, activation_map_list = system.call_classifier(images)
        >>>     return activation_list, activation_map_list
        """
        score_list = []
        out_images = []
        for image in image_list:
            if  image==None: # for dalle
                score_list.append(None)
                out_images.append(None)
            else:
                image = str2image(image)
                act, resized_image = self.classifier.calc_score(image)
                score_list.append(act)
                out_images.append(image2str(resized_image))

        return score_list, out_images

class Tools:
    """
    A Python class containing tools to interact with the units implemented in the system class, 
    in order to run experiments on it.
    
    Attributes
    ----------
    text2image_model_name : str
        The name of the text-to-image model.
    text2image_model : any
        The loaded text-to-image model.
    threshold : any
        Activation threshold for classifier analysis.
    device : torch.device
        The device (CPU/GPU) used for computations.
    experiment_log: str
        A log of all the experiments, including the code and the output from the classifier
        analysis.
    exemplars : Dict
        A dictionary containing the exemplar images for each unit.
    exemplars_activations : Dict
        A dictionary containing the activations for each exemplar image.
    exemplars_thresholds : Dict
        A dictionary containing the threshold values for each unit.
    results_list : List
        A list of the results from the classifier analysis.


    Methods
    -------
    dataset_exemplars(self, unit_ids: List[int], system: System)->List[Tuple[float, str]]]:
        Retrieves the activations and exemplar images for a list of units.
    edit_images(self, image_prompts : List[Image.Image], editing_prompts : List[str]):
        Generate images from a list of prompts, then edits each image with the
        corresponding editing prompt.
    text2image(self, prompt_list: List[str]) -> List[torch.Tensor]:
        Gets a list of text prompt as an input, generates an image for each prompt in the list using a text to image model.
        The function returns a list of images.
    summarize_images(self, image_list: List[str]) -> str:
        Gets a list of images and describes what is common to all of them, focusing specifically on unmasked regions.
    sampler(act: any, imgs: List[any], mask: any, prompt: str, method: str = 'max') -> Tuple[List[int], List[str]]
        Processes images based on classifier activations.
    describe_images(self, image_list: List[str], image_title:List[str]) -> str:
        Generates descriptions for a list of images, focusing specifically on highlighted regions.
    display(self, *args: Union[str, Image.Image]):
        Displays a series of images and/or text in the chat, similar to a Jupyter notebook.

    """

    def __init__(self, device: str, SyntheticExemplars: SyntheticExemplars = None, text2image_model_name='sd', p2p_model_name='ip2p', image2text_model_name='gpt-4o'):
        """
        Initializes the Tools object.

        Parameters
        ----------
        device : str
            The computational device ('cpu' or 'cuda').
        SyntheticExemplars : object
            an object from the class SyntheticExemplars
        text2image_model_name : str
            The name of the text-to-image model.
        p2p_model_name : str
            The name of the p2p model.
        """
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.image2text_model_name = image2text_model_name
        self.text2image_model_name = text2image_model_name
        self.text2image_model = self._load_text2image_model(model_name=text2image_model_name)
        self.p2p_model_name = p2p_model_name
        if self.p2p_model_name == 'ip2p':
            self.p2p_model = self._load_pix2pix_model(model_name=self.p2p_model_name) # consider maybe adding options for other models like pix2pix zero
        elif self.p2p_model_name == "instdiff":
            self.p2p_model = InstructDiffusion(batch_size=1, config_path="utils/InstructDiffusion/configs/instruct_diffusion.yaml", model_path="utils/InstructDiffusion/checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt", device=self.device)
        self.experiment_log = []
        self.im_size = 224
        if SyntheticExemplars:
            self.exemplars = SyntheticExemplars.exemplars
            self.exemplars_scores = SyntheticExemplars.scores
        else:
            self.exemplars = None
            self.exemplars_scores = None
        self.score_threshold = 0
        self.results_list = []

    def dataset_exemplars(self, system: Synthetic_System)->List[Tuple[float, str]]:
        """
        Retrieves the activations and exemplar images the specified units.

       Parameters
        ----------
        system : System
            The system representing the specific classifier and layer within the neural network.
            The system should have 'layer' and 'classifier_num' attributes, so the dataset_exemplars function 
            can return the exemplar activations and masked images for that specific classifier.

        Returns
        -------
        List[Tuple[float, str]]
            For each exemplar image, stores a tuple containing two elements:
            - The first element is the activation value for the specified classifier.
            - The second element is the exemplar images (as Base64 encoded strings) corresponding to the activation.

        Example
        -------
        >>> # Display the exemplars and activations for a list of units
        >>> unit_ids = [0, 1]
        >>> exemplar_data = tools.dataset_exemplars(unit_ids, system)
        >>> for i in range(len(exemplar_data)):
        >>>     tools.display(f"unit {unit_ids[i]}: ")
        >>>     for activation, masked_image in exemplar_data[i]:
        >>>         tools.display(masked_image, activation)
        """
        image_list = self.exemplars[system.layer][system.classifier_num]
        score_list = self.exemplars_scores[system.layer][system.classifier_num]
        self.score_threshold = sum(score_list)/len(score_list)
        score_list = [round(score, 4) for score in score_list]
        return list(zip(score_list, image_list))

    def edit_images(self,
                    base_images: List[str],
                    editing_prompts: List[str]) -> Tuple[List[str], List[str]]:
        """
        Generates or uses provided base images, then edits each base image with a
        corresponding editing prompt. Accepts either text prompts or Base64
        encoded strings as sources for the base images.

        The function returns a list containing lists of images (original and edited,
        interleaved) in Base64 encoded string format, and a list of the relevant
        prompts (original source string and editing prompt, interleaved).

        Parameters
        ----------
        base_images : List[str]
            A list of images as Base64 encoded strings. These images are to be 
            edited by the prompts in editing_prompts.
        editing_prompts : List[str]
            A list of instructions for how to edit the base images derived from
            `base_images`. Must be the same length as `base_images`.

        Returns
        -------
        Tuple[List[str]], List[str]]
            - all_images: A list where elements alternate between:
                - A list of Base64 strings for the original image(s) from a source.
                - A list of Base64 strings for the edited image(s) from that source.
              Example: [[orig1_img1, orig1_img2], [edit1_img1, edit1_img2], [orig2_img1], [edit2_img1], ...]
            - all_prompts: A list where elements alternate between:
                - The original source string (text prompt or Base64) used.
                - The editing prompt used.
              Example: [source1, edit1, source2, edit2, ...]
            The order in `all_images` corresponds to the order in `all_prompts`.

        Raises
        ------
        ValueError
            If the lengths of `base_images` and `editing_prompts` are not equal.

        Examples
        --------
        >>> # test the confidence score of the classifier for the prompt "a dog standing on the grass"
        >>> # for the same image but with different actions instead of "standing":
        >>> prompts = ["a landscape with a tree and a river"]*3
        >>> original_images = tools.text2image(prompts)
        >>> edits = ["make it autumn","make it spring","make it winter"]
        >>> all_images, all_prompts = tools.edit_images(original_images, edits)
        >>> score_list, image_list = system.call_classifier(all_images)
        >>> for score, image, prompt in zip(score_list, image_list, all_prompts):
        >>>     tools.display(image, f"Prompt: {prompt}\nConfidence Score: {score}")
        >>> 
        >>> # test the confidence score of the classifier on the highest scoring dataset exemplar
        >>> # under different conditions        
        >>> exemplar_data = tools.dataset_exemplars(system)
        >>> highest_scoring_exemplar = exemplar_data[0][1]
        >>> edits = ["make it night","make it daytime","make it snowing"]
        >>> all_images, all_prompts = tools.edit_images([highest_scoring_exemplar]*len(edits), edits)
        >>> score_list, image_list = system.call_classifier(all_images)
        >>> for score, image, prompt in zip(score_list, image_list, all_prompts):
        >>>     tools.display(image, f"Prompt: {prompt}\nConfidence Score: {score}")
        """
        if len(base_images) != len(editing_prompts):
            raise ValueError("Length of base_images and editing_prompts must be equal.")

        edited_images_b64_lists = []
        base_imgs_obj = [str2image(img_b64) for img_b64 in base_images]

        for i in range(len(base_images)):
            if self.p2p_model_name == "instdiff":
                # Model returns a list of image objects
                edited_imgs_obj = self.p2p_model([editing_prompts[i]], [base_imgs_obj[i]])
            else:
                # Model returns an object with an 'images' attribute (list of image objects)
                result = self.p2p_model([editing_prompts[i]], [base_imgs_obj[i]])
                edited_imgs_obj = result.images
            edited_images_b64_lists.append(image2str(edited_imgs_obj[0]))

        # --- Interleave Results ---
        all_images = []
        all_prompts = []
        for i in range(len(base_images)):
            # Add original image(s) list and the original source string
            all_images.append(base_images[i])
            all_prompts.append("Original Image")

            # Add edited image(s) list and the editing prompt
            all_images.append(edited_images_b64_lists[i])
            all_prompts.append(f"Editing Prompt: {editing_prompts[i]}")

        return all_images, all_prompts
        
    def text2image(self, prompt_list: List[str]) -> List[str]:
        """Gets a list of text prompt as an input, generates an image for each prompt in the list using a text to image model.
        The function returns a list of images.

        Parameters
        ----------
        prompt_list : List[str]
            A list of text prompts for image generation.

        Returns
        -------
        List[Image.Image]
            A list of images, corresponding to each of the input prompts. 


        Examples
        --------
        >>> # Generate images from a list of prompts
        >>>     prompt_list = [“a toilet on mars”, 
        >>>                     “a toilet on venus”,
        >>>                     “a toilet on pluto”]
        >>>     images = tools.text2image(prompt_list)
        >>>     tools.display(*images)
        """
        image_list = [] 
        for prompt in prompt_list:
            image = self._prompt2image(prompt)
            image_list.append(image)
        return image_list
    
 
    def summarize_images(self, image_list: List[str]) -> str:
        """
        Gets a list of images and describes what is common to all of them, focusing specifically on unmasked regions.


        Parameters
        ----------
        image_list : list
            A list of images in Base64 encoded string format.
        
        Returns
        -------
        str
            A string with a descriptions of what is common to all the images.

        Example
        -------
        >>> # Summarize a unit's dataset exemplars
        >>> _, exemplars = tools.dataset_exemplars([0], system)[0] # Get exemplars for unit 0
        >>> summarization = tools.summarize_images(image_list)
        >>> tools.display("Unit 0 summarization: ", summarization)
        >>> 
        >>> # Summarize what's common amongst two sets of exemplars
        >>> exemplars_data = tools.dataset_exemplars([0,1], system)
        >>> all_exemplars = []
        >>> for _, exemplars in exemplars_data:
        >>>     all_exemplars += exemplars
        >>> summarization = tools.summarize_images(all_exemplars)
        >>> tools.display("All exemplars summarization: ", summarization)
        First, list the common features that you see in the regions such as:

        Non-semantic Concepts (Shape, texture, color, etc.): ...
        Semantic Concepts (Objects, animals, people, scenes, etc): ...
        """
        image_list = self._description_helper(image_list)
        instructions = "What do all of these images have in common? There might be more than one common concept, or a few groups of images each with different common concepts. In these cases return all of the concepts. Return your description in the following format: [COMMON]: <your description>."
        history = [{'role': 'system', 
                    'content': 
                        'You are a helpful assistant who views/compares images.'}]
        user_content = [{"type":"text", "text": instructions}]
        for ind,image in enumerate(image_list):
            user_content.append(format_api_content("image_url", image))
        history.append({'role': 'user', 'content': user_content})
        description = ask_agent(self.image2text_model_name,history)
        if isinstance(description, Exception): return description
        return description

    def describe_images(self, image_list: List[str], image_title:List[str]) -> str:
        """
        Provides impartial description of the highlighted image regions within an image.
        Generates textual descriptions for a list of images, focusing specifically on highlighted regions.
        This function translates the visual content of the highlighted region in the image to a text description. 
        The function operates independently of the current hypothesis list and thus offers an impartial description of the visual content.        
        It iterates through a list of images, requesting a description for the 
        highlighted (unmasked) regions in each synthetic image. The final descriptions are concatenated 
        and returned as a single string, with each description associated with the corresponding 
        image title.

        Parameters
        ----------
        image_list : List[str]
            A list of images in Base64 encoded string format.
        image_title : List[str]
            A list of titles for each image in the image_list.

        Returns
        -------
        str
            A concatenated string of descriptions for each image, where each description 
            is associated with the image's title and focuses on the highlighted regions 
            in the image.

        Example
        -------
        >>> prompt_list = [“a man with two teeth”, 
                            “a man with fangs”,
                            “a man with all molars, like a horse”]
        >>> images = tools.text2image(prompt_list)
        >>> activation_list, masked_images = system.units(images, [0])[0]
        >>> descriptions = tools.describe_images(masked_images, prompt_list)
        >>> tools.display(*descriptions)
        """
        description_list = ''
        instructions = "Please describe the image as concisely as possible. Return your description in the following format: [Description]: <your concise description>"
        time.sleep(60)
        image_list = self._description_helper(image_list)
        for ind,image in enumerate(image_list):
            history = [{'role':'system', 
                        'content':'you are an helpful assistant'},
                        {'role': 'user', 
                         'content': 
                         [format_api_content("text", instructions),
                           format_api_content("image_url", image)]}]
            description = ask_agent(self.image2text_model_name,history)
            if isinstance(description, Exception): return description_list
            description = description.split("[highlighted regions]:")[-1]
            description = " ".join([f'"{image_title[ind]}", highlighted regions:',description])
            description_list += description + '\n'
        return description_list

    def _description_helper(self, *args: Union[str, Image.Image]):
        '''Helper function for display to recursively handle iterable arguments.'''
        output = []
        for item in args:
            if isinstance(item, (list, tuple)):
                output.extend(self._description_helper(*item))
            else:
                output.append(item)
        return output
    
    # TODO - Document new style
    def display(self, *args: Union[str, Image.Image]):
        """
        Displays a series of images and/or text in the chat, similar to a Jupyter notebook.
        
        Parameters
        ----------
        *args : Union[str, Image.Image]
            The content to be displayed in the chat. Can be multiple strings or Image objects.

        Notes
        -------
        Displays directly to chat interface.

        Example
        -------
        >>> # Display a single image
        >>> prompt = ["a dog standing on the grass"]
        >>> images = tools.text2image(prompt)
        >>> tools.display(*images)
        >>>
        >>> # Display a list of images
        >>> prompt_list = ["A green creature",
        >>>                 "A red creature",
        >>>                 "A blue creature"]
        >>> images = tools.text2image(prompt_list)
        >>> tools.display(*images)
        """
        output = []
        for item in args:
            # Check if tuple or list
            if isinstance(item, (list, tuple)):
                output.extend(self._display_helper(*item))
            else:
                output.append(self._process_chat_input(item))
        self.update_experiment_log(role='user', content=output)
    
    def _display_helper(self, *args: Union[str, Image.Image]):
        '''Helper function for display to recursively handle iterable arguments.'''
        output = []
        for item in args:
            if isinstance(item, (list, tuple)):
                output.extend(self._display_helper(*item))
            else:
                output.append(self._process_chat_input(item))
        return output

    def update_experiment_log(self, role, content=None, type=None, type_content=None):
        openai_role = {'execution':'user','agent':'assistant','user':'user','system':'system'}
        if type == None:
            self.experiment_log.append({'role': openai_role[role], 'content': content})
        elif content == None:
            if type == 'text':
                self.experiment_log.append({'role': openai_role[role], 'content': [{"type":type, "text": type_content}]})
            if type == 'image_url':
                self.experiment_log.append({'role': openai_role[role], 'content': [{"type":type, "image_url": type_content}]})

    def _process_chat_input(self, content: Union[str, Image.Image]) -> Dict[str, str]:
        '''Processes the input content for the chatbot.
        
        Parameters
        ----------
        content : Union[str, Image.Image]
            The input content to be processed.'''

        if is_base64(content):
            return format_api_content("image_url", content)
        elif isinstance(content, Image.Image):
            return format_api_content("image_url", image2str(content))
        else:
            return format_api_content("text", content)

    def _load_pix2pix_model(self, model_name):
        """
        Loads a pix2pix image editing model.

        Parameters
        ----------
        model_name : str
            The name of the pix2pix model.

        Returns
        -------
        The loaded pix2pix model.
        """
        if model_name == "ip2p": # instruction tuned pix2pix model
            print(f"Loading InstructPix2Pix model from timbrooks/instruct-pix2pix...")
            device = self.device
            model_id = "timbrooks/instruct-pix2pix"
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
            pipe = pipe.to(device)
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

            # Set progress bar to quiet mode
            pipe.set_progress_bar_config(disable=True)
            return pipe
        else:
            raise("unrecognized pix2pix model name")

    
    def _load_text2image_model(self,model_name):
        """
        Loads a text-to-image model.

        Parameter
        ----------
        model_name : str
            The name of the text-to-image model.

        Returns
        -------
        The loaded text-to-image model.
        """
        if model_name == "sd":
            print(f"Loading Stable Diffusion 3.5 Medium model from stabilityai/stable-diffusion-3.5-medium...")
            device = self.device
            model_id = "stabilityai/stable-diffusion-3.5-medium"
            sdpipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
            sdpipe = sdpipe.to(device)

            # TODO - Testing, then add to others or remove
            # Set progress bar to quiet mode
            sdpipe.set_progress_bar_config(disable=True)

            return sdpipe
        elif model_name == "sdxl-turbo":
            print(f"Loading SDXL Turbo model from stabilityai/sdxl-turbo...")
            device = self.device
            model_id = "stabilityai/sdxl-turbo"
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
            pipe = pipe.to(device)
            return pipe
        elif model_name == "dalle":
            pipe = None
            return pipe
        elif model_name == "flux":
            print(f"Loading Flux Image Generator...")
            return FluxImageGenerator(device=self.device)
        else:
            raise("unrecognized text to image model name")
    
    def _prompt2image(self, prompt):
        if self.text2image_model_name == "sd":
            image = self.text2image_model(prompt).images[0]
        elif self.text2image_model_name == "flux":
            image = self.text2image_model(prompt)
            
        elif self.text2image_model_name == "dalle":
            try:
                prompt = "a photo-realistic image of " + prompt
                response = openai.Image.create(prompt=prompt, 
                                            model="dall-e-3",
                                            n=1, 
                                            size="1024x1024",
                                            quality="hd",
                                            response_format="b64_json"
                                            )
                image = response.data[0].b64_json
                image = str2image(image)
            except Exception as e:
                raise(e)
        image = image.resize((self.im_size, self.im_size))
        return image2str(image)

    def _generate_safe_images(self, prompts: List[str], max_attempts:int = 10):
        results = []
        for prompt in prompts:
            safe_image = self._generate_single_safe_image(prompt, max_attempts)
            results.append(safe_image)
        return results
    
    def _generate_single_safe_image(self, prompt, max_attempts):
        for attempt in range(max_attempts):
            # Generate the image
            result = self.text2image_model(prompt)
            
            # Check if the image is safe (not NSFW)
            if not result.nsfw_content_detected[0]:
                return result.images[0]  # Return the safe image
            
            print(f"Prompt '{prompt}': Attempt {attempt + 1}: NSFW content detected. Retrying...")
        
        raise Exception(f"Prompt '{prompt}': Failed to generate a safe image after {max_attempts} attempts")
    
    def generate_html(self, path2save, name="experiment", line_length=100):
        # Generates an HTML file with the experiment log.
        html_string = f'''<html>
        <head>
        <title>Experiment Log</title>
        <!-- Include Prism Core CSS (Choose the theme you prefer) -->
        <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/themes/prism.min.css" rel="stylesheet" />
        <!-- Include Prism Core JavaScript -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/prism.min.js"></script>
        <!-- Include the Python language component for Prism -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/components/prism-python.min.js"></script>
        </head> 
        <body>
        <h1>{path2save}</h1>'''

        # don't plot system+user prompts (uncomment if you want the html to include the system+user prompts)
        '''
        html_string += f"<h2>{self.experiment_log[0]['role']}</h2>"
        html_string += f"<pre><code>{self.experiment_log[0]['content']}</code></pre><br>"
        
        html_string += f"<h2>{self.experiment_log[1]['role']}</h2>"
        html_string += f"<pre>{self.experiment_log[1]['content'][0]}</pre><br>"
        initial_images = ''
        initial_activations = ''
        for cont in self.experiment_log[1]['content'][1:]:
            if isinstance(cont, dict):
                initial_images += f"<img src="data:image/png;base64,{cont['image']}"/>"
            else:
                initial_activations += f"{cont}    "
        html_string +=  initial_images
        html_string += f"<p>Activations:</p>"
        html_string += initial_activations
        '''
        for entry in self.experiment_log:      
            if entry['role'] == 'assistant':
                html_string += f"<h2>MAIA</h2>"  
                text = entry['content'][0]['text']
                # Wrap text to line_length
                #text = textwrap.fill(text, line_length)

                html_string += f"<pre>{text}</pre><br>"
                html_string += f"<h2>Experiment Execution</h2>"  
            else:
                for content_entry in entry['content']:

                    if "image_url" in content_entry["type"]:
                        html_string += f'''<img src="{content_entry['image_url']['url']}"/>'''  
                    elif "text" in content_entry["type"]:
                        html_string += f"<pre>{content_entry['text']}</pre>"
        html_string += '</body></html>'

        # Save
        file_path = os.path.join(path2save, f"{name}.html")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(html_string)
            
        
        
