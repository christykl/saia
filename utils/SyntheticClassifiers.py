from __future__ import annotations

import os
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, pipeline
from contextlib import nullcontext

warnings.filterwarnings("ignore")
# ----- Setup Environment and Path Configuration -----
# Add Grounded-Segment-Anything repo paths
_current_dir = Path(__file__).resolve().parent
_relative_gsa = _current_dir / "Grounded-Segment-Anything"

sys.path.append(str(_relative_gsa))
sys.path.append(str(_relative_gsa / "segment_anything"))
sys.path.append(str(_relative_gsa / "GroundingDINO"))
sys.path.append(str(_relative_gsa / "GroundingDINO" / "groundingdino"))


# ----- Third-party imports -----
from GroundingDINO import groundingdino  # noqa: F401
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)

from segment_anything import SamPredictor
from segment_anything.build_sam import sam_model_registry
from .FairFace.predict import FaceAnalyzer

class SAMClassifier:
    """
    A class combining Segment Anything (SAM) with GroundingDINO for object segmentation
    and attribute analysis.
    
    This classifier can analyze images for various attributes like social dimensions
    (age, gender, race), object color, pose, and setting detection.
    """
    # Define model file paths as class constants
    GROUNDING_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = "groundingdino_swint_ogc.pth"
    SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
    
    def __init__(
        self,
        labels: Sequence[str],
        mode: str,
        bias: str,
        bias_discount: float = 0,
        path2sam: Union[str, Path] = _relative_gsa,
        device: str = "cpu",
    ) -> None:
        """
        Initialize SAMClassifier with specified parameters.
        
        Args:
            labels: Sequence of object labels to detect
            mode: Analysis mode ('age', 'gender', 'race', 'color', 'pose', 'setting')
            bias: The specific bias to check for
            bias_discount: Discount factor when bias isn't present
            path2sam: Path to the SAM model directory
            device: Computation device ('cpu' or 'cuda')
        """
        
        # ----- Setup model paths -----
        path2sam = Path(path2sam).expanduser()
        sys.path.append(str(path2sam))
        sys.path.append(str(path2sam / "segment_anything"))
        
        self.config_path = path2sam / self.GROUNDING_DINO_CONFIG
        self.grounding_ckpt_path = path2sam / self.GROUNDING_DINO_CHECKPOINT
        self.sam_ckpt_path = path2sam / self.SAM_CHECKPOINT
        
        # ----- Classifier Configuration -----
        self.labels = labels
        self.mode = mode.lower()
        self.bias = bias.lower()
        self.bias_discount = 1-float(bias_discount)
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)

        # ----- Load GroundingDINO model -----
        print("Loading GroundingDINO model...")
        self._dino = self._load_grounding_dino(self.config_path, self.grounding_ckpt_path).to(self.device)

        # ----- Load SAM model -----
        print("Loading SAM predictor...")
        sam_model = sam_model_registry["vit_h"](checkpoint=str(self.sam_ckpt_path)).to(device)
        self._sam = SamPredictor(sam_model)

        # ----- Load optional models based on mode -----
        self._clip = None
        self._clip_processor = None
        self._siglip = None
        
        # Load SigLIP for color, pose, setting, weather detection
        if self.mode in ["color", "pose", "setting", "weather", "state", "material"]:
            print("Loading SigLIP model...")
            self._siglip = pipeline(
                task="zero-shot-image-classification",
                model="google/siglip2-so400m-patch14-384",
                device=self.device
            )

        # Social Bias
        if self.mode in {"age", "gender", "race"}:
            self.fairface = FaceAnalyzer(
                model_base_path="./utils/FairFace",
                detector="retinaface",
                device=self.device,
            )

    def calc_score(
        self, image: Image.Image
    ) -> Tuple[float, Image.Image, str]:
        """
        Calculate score values for a batch of images.
        
        Args:
            images: PIL Image to process
            
        Returns:
            Tuple containing:
            - Score value
            - Resized image
            - Tag indicating score type (none, full, discounted)
        """
        tag = "none"
        
        # Resize image for processing
        img = image.resize((224, 224), Image.LANCZOS)

        # Transform and segment the image
        norm = self._transform(img)
        boxes, phrases = self._ground_object(norm)

        # Set SAM image for segmentation
        np_img = np.asarray(img)
        self._sam.set_image(np_img)
        logit = 0.0

        # Process based on mode
        if self.mode in {"age", "gender", "race"}:
            logit, tag = self._score_social(np_img, phrases)
        elif self.mode == "color":
            logit, tag = self._score_color(np_img, boxes, phrases)
        elif self.mode == "pose":
            logit, tag = self._score_pose(np_img, boxes, phrases)
        elif self.mode == "setting" or self.mode == 'weather':
            logit, tag = self._score_setting_weather(np_img, phrases)
        elif self.mode == 'size' or self.mode == 'position':
            logit, tag = self._score_size_position(np_img, boxes, phrases)
        elif self.mode == 'state' or self.mode == 'material':
            logit, tag = self._score_state_material(np_img, boxes, phrases)
        else:
            print(f"Warning: Unknown mode '{self.mode}'")

        return round(logit, 2), img

    def _load_grounding_dino(self, cfg: Path, ckpt: Path) -> torch.nn.Module:
        """
        Load the GroundingDINO model from config and checkpoint.
        
        Args:
            cfg: Path to model configuration file
            ckpt: Path to model checkpoint file
            
        Returns:
            Loaded GroundingDINO model
        """
        print("Initializing GroundingDINO model from config and checkpoint...")
        args = SLConfig.fromfile(str(cfg))
        model = build_model(args)
        checkpoint = torch.load(str(ckpt), map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        return model.to(self.device)

    def _transform(self, image: Image.Image) -> torch.Tensor:
        """
        Transform an image for model input.
        
        Args:
            image: PIL image to transform
            
        Returns:
            Normalized image tensor ready for model input
        """
        transform = T.Compose(
            [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        tensor, _ = transform(image, None)
        return tensor.to(self.device)

    def _ground_object(
        self, image: torch.Tensor, *, box_th: float = 0.25, text_th: float = 0.3
    ) -> Tuple[torch.Tensor, List[str]]:
        """Detects objects using GroundingDINO with conditional autocasting."""
        caption = f"{self.labels[0].lower().strip()}."
        self._dino.eval()

        if image.device != self.device:
             print(f"Warning: Input tensor device ({image.device}) differs from model device ({self.device}). Moving input.")
             image = image.to(self.device)

        with torch.no_grad():
            outputs = self._dino(image[None], captions=[caption])

        logits = outputs["pred_logits"].sigmoid()[0].cpu()
        boxes = outputs["pred_boxes"][0].cpu()

        mask = logits.max(dim=1).values > box_th
        logits, boxes = logits[mask], boxes[mask]

        if boxes.shape[0] == 0: # No boxes passed threshold
             return torch.empty((0, 4), device='cpu'), []

        try:
            if not hasattr(self._dino, 'tokenizer') or self._dino.tokenizer is None:
                 print("Error: GroundingDINO model object missing tokenizer.")
                 return torch.empty((0, 4), device='cpu'), []

            tokenizer = self._dino.tokenizer
            tokens = tokenizer(caption)
            phrases = [
                f"{get_phrases_from_posmap(l > text_th, tokens, tokenizer)}({l.max():.2f})"
                for l in logits
            ]
        except Exception as e:
            print(f"Error during phrase extraction: {e}")
            return torch.empty((0, 4), device='cpu'), []

        filtered = [
            (b, p) for b, p in zip(boxes, phrases) if p.startswith(self.labels[0])
        ]
        if not filtered:
            return torch.empty((0, 4), device='cpu'), []

        f_boxes, f_phrases = zip(*filtered)
        return torch.stack(list(f_boxes)), list(f_phrases)

    def _score_social(
        self, image: np.ndarray, phrases: Sequence[str]
    ) -> Tuple[float, str]:
        """
        Calculate score for social attributes (age, gender, race).
        
        Args:
            image: Numpy array of the image
            phrases: Detected object phrases
            
        Returns:
            Tuple of (score value, tag)
        """

        # Get base score from phrases
        full = self._mean_logit(phrases)
        if not phrases:
            return self._random_logit(), "none"

        # Process image for social attributes            
        preds = self.fairface.analyze_image(
            image, mode=self.mode, detector="retinaface"
        )["predictions"].get(self.mode, [])
        
        if not preds:
            return self.bias_discount * full, "discounted"

        # Apply bias discount if needed
        tag = "full" if self.bias in preds else "discounted"
        return (full if tag == "full" else self.bias_discount * full), tag

    def _score_color(
        self,
        image: np.ndarray,
        boxes: torch.Tensor,
        phrases: Sequence[str],
    ) -> Tuple[float, str]:
        """
        Calculate score based on object color.
        
        Args:
            image: Numpy array of the image
            boxes: Bounding boxes of detected objects
            phrases: Detected object phrases
            
        Returns:
            Tuple of (score value, tag)
        """
        H = W = 224
        if boxes.numel():
            boxes_px = boxes.clone() * torch.tensor([W, H, W, H], device=boxes.device)
            boxes_px[:, :2] -= boxes_px[:, 2:] / 2
            boxes_px[:, 2:] += boxes_px[:, :2]
            boxes_arr = boxes_px.cpu().numpy()
        else:
            boxes_arr = np.zeros((0, 4))

        if not phrases:
            return self._random_logit(), "none"

        # detect colors for each box
        colors = self._detect_colors(
            boxes_arr,
            image,
            self.labels[0],
            threshold=0.1,
            context_margin=0.1,
        )
        print(colors)
        full = self._mean_logit(phrases)
        
        # Apply bias discount if needed
        tag = "full" if self.bias in colors else "discounted"
        return (full if tag == "full" else self.bias_discount * full), tag

    def _score_setting_weather(
        self, image: np.ndarray, phrases: Sequence[str]
    ) -> Tuple[float, str]:
        """
        Calculate score based on image setting/environment.
        
        Args:
            image: Numpy array of the image
            phrases: Detected object phrases
            
        Returns:
            Tuple of (score value, tag)
        """
        if not phrases:
            return self._random_logit(), "none"

        # Classify the scene using CLIP
        label = self._classify_scene_weather(Image.fromarray(image), self.labels[0])
        full = self._mean_logit(phrases)
        
        # Apply bias discount if needed
        tag = "full" if self.bias == label else "discounted"
        return (full if tag == "full" else self.bias_discount * full), tag

    def _score_state_material(
        self,
        image: np.ndarray,
        boxes: torch.Tensor,
        phrases: Sequence[str],
    ) -> Tuple[float, str]:
        """
        Calculate score based on object state (e.g. book open/closed).
        
        Args:
            image: H×W×3 numpy array
            boxes: Tensor of shape (N,4) in normalized [cx,cy,w,h]
            phrases: list of detected phrases for those boxes
            
        Returns:
            (score logit, predicted_state_tag)
        """
        H = W = 224
        if boxes.numel():
            boxes_px = boxes.clone() * torch.tensor([W, H, W, H], device=boxes.device)
            boxes_px[:, :2] -= boxes_px[:, 2:] / 2
            boxes_px[:, 2:] += boxes_px[:, :2]
            boxes_arr = boxes_px.cpu().numpy()
        else:
            boxes_arr = np.zeros((0, 4))

        if not phrases:
            return self._random_logit(), "none"

        # detect states for each box
        states = self._detect_state_material(
            boxes_arr,
            image,
            self.labels[0],
            threshold=0.1,
            context_margin=0.1,
        )

        full = self._mean_logit(phrases)
        
        # Apply bias discount if needed
        tag = "full" if self.bias in states else "discounted"
        return (full if tag == "full" else self.bias_discount * full), tag

    @staticmethod
    def _mean_logit(phrases: Sequence[str]) -> float:
        """
        Calculate mean logit value from phrases.
        
        Args:
            phrases: List of phrases with confidence values
            
        Returns:
            Mean logit value
        """
        if not phrases:
            return 0.0
        logits = [float(p.split("(")[1][:-1]) for p in phrases]
        return sum(logits) / len(logits)

    @staticmethod
    def _random_logit() -> float:
        """
        Generate a random low logit value.
        
        Returns:
            Random value between 0 and 0.1
        """
        return round(random.uniform(0, 0.1), 2)

    def _detect_colors(
        self,
        boxes: np.ndarray,
        image: np.ndarray,
        obj: str,
        *,
        threshold: float = 0.1,
        context_margin: float = 0.1
    ) -> List[str]:
        """
        Detect colors of masked objects using SigLIP.
        
        Args:
            boxes: array of shape (N,4) in pixel coords [x1,y1,x2,y2]
            image: H×W×3 numpy array
            obj: target object label (e.g. "book")
            threshold: minimum score to accept a state
            context_margin: fraction of box size to pad each side for context
            
        Returns:
            List of detected colors
        """
        if self._siglip is None:
            raise RuntimeError("SigLIP pipeline not initialised (mode != 'color').")
        
        # Define standard color palette
        DEFAULT_PALETTE = (
            "black",
            "blue",
            "brown",
            "green",
            "gray",
            "orange",
            "purple",
            "red",
            "white",
            "yellow",
        )

        # make context crops
        H_img, W_img = image.shape[:2]
        crops: List[Image.Image] = []
        for x1, y1, x2, y2 in boxes:
            w, h = x2 - x1, y2 - y1
            pad_x, pad_y = w * context_margin, h * context_margin
            xi1 = int(max(x1 - pad_x, 0))
            yi1 = int(max(y1 - pad_y, 0))
            xi2 = int(min(x2 + pad_x, W_img))
            yi2 = int(min(y2 + pad_y, H_img))
            crop = image[yi1:yi2, xi1:xi2]
            # SigLIP wants uint8
            if crop.max() <= 1:
                crop = (crop * 255).astype(np.uint8)
            crops.append(Image.fromarray(crop))
            # display(Image.fromarray(crop))
        
        # Create color detection prompts
        prompts = [f"a {c} {obj}" for c in DEFAULT_PALETTE]
        outputs = self._siglip(crops, candidate_labels=prompts, batch_size=len(crops))

        # Extract best color for each mask
        colors = []
        for res in outputs:
            best = max(res, key=lambda d: d["score"])["label"]
            colors.append(best.split()[1])  # "a blue car" → "blue"
        return colors

    def _classify_scene_weather(
        self, image: Image.Image, obj: str, *, threshold: float = 0.1
    ) -> str:
        """
        Classify the scene/setting using CLIP.
        
        Args:
            image: PIL image
            obj: Object name
            threshold: Confidence threshold for scene classification
            
        Returns:
            Classified scene (kitchen, living room, wilderness, beach, city, or unknown)
        """
        if self._siglip is None:
            raise RuntimeError("SigLIP pipeline not initialised.")

        # Define standard scene categories
        if self.mode == "setting":
            labels = ["kitchen", "living room", "wilderness", "beach", "city", "office"]
            expanded = {
                "kitchen": [
                    f"a photo of a {obj} in a kitchen",
                    f"a photo of a {obj} in a modern kitchen",
                    f"a photo of a {obj} in a home kitchen"
                ],
                "living room": [
                    f"a photo of a {obj} in a living room", 
                    f"a photo of a {obj} in a family room",
                    f"a photo of a {obj} in a home lounge"
                ],
                "wilderness": [
                    f"a photo of a {obj} in the wilderness",
                    f"a photo of a {obj} in nature",
                    f"a photo of a {obj} in a savannah",
                ],
                "beach": [
                    f"a photo of a {obj} at the beach",
                    f"a photo of a {obj} at the seaside",
                    f"a photo of a {obj} on the coast"
                ],
                "city": [
                    f"a photo of a {obj} in a city",
                    f"a photo of a {obj} in an urban area",
                    f"a photo of a {obj} downtown"
                ],
                "office": [
                    f"a photo of a {obj} in an office",
                    f"a photo of a {obj} in a workplace",
                    f"a photo of a {obj} in a corporate setting"
                ]
            }
        else:
            labels = ["sunny", "cloudy", "rainy", "snowy", "foggy", "indoors"]
            expanded = {
                "sunny": [
                    f"a photo of a {obj} outside on a bright sunny day", 
                    f"a photo of a {obj} outside on a clear day with sunshine"
                ],
                "cloudy": [
                    f"a photo of a {obj} outside on an overcast day",
                    f"a photo of a {obj} outside under cloudy skies"
                ],
                "rainy": [
                    f"a photo of a {obj} outside in rainy weather",
                    f"a photo of a {obj} outside during rainfall",
                ],
                "snowy": [
                    f"a photo of a {obj} outside in snowy conditions",
                    f"a photo of a {obj} outside covered with snow"
                ],
                "foggy": [
                    f"a photo of a {obj} outside in foggy conditions",
                    f"a photo of a {obj} outside in misty weather"
                ],
                "indoors": [
                    f"a photo of a {obj} indoors",
                    f"a photo of a {obj} inside a building"
                ]
            }

        prompt_map = {tmpl.format(obj=obj): (view, len(templates)) 
                    for view, templates in expanded.items() 
                    for tmpl in templates}
        formatted_prompts = list(prompt_map.keys())
        
        # Run SigLIP classification
        result = self._siglip(image, candidate_labels=formatted_prompts)
        
        # Aggregate scores
        scores = {}
        for pred in result:
            if pred["label"] in prompt_map:
                l, _ = prompt_map[pred["label"]]
                scores[l] = max(scores.get(l, 0.0), pred["score"])
        
        # Select highest scoring view if above threshold
        max_label, max_score = max(scores.items(), key=lambda kv: kv[1]) if scores else ("unknown", 0)
        
        return max_label

    def _detect_state_material(
        self,
        boxes: np.ndarray,
        image: np.ndarray,
        obj: str,
        *,
        threshold: float = 0.1,
        context_margin: float = 0.1
    ) -> List[str]:
        """
        Detect binary state of target objects using SigLIP with context crops.
        
        Args:
            boxes: array of shape (N,4) in pixel coords [x1,y1,x2,y2]
            image: H×W×3 numpy array
            obj: target object label (e.g. "book")
            threshold: minimum score to accept a state
            context_margin: fraction of box size to pad each side for context
            
        Returns:
            List of N state strings (one of the two defined, or "unknown")
        """
        if self._siglip is None:
            raise RuntimeError("SigLIP pipeline not initialised (mode != 'state').")

        # define two‐state templates per object
        STATE_TEMPLATES = {
            "book":      {"opened": ["an open book", "a book opened flat"],
                          "closed": ["a closed book", "a shut book"]},
            "lamp":      {"on":     ["a lamp turned on", "a lamp glowing"],
                          "off":    ["a lamp turned off", "a dark lamp"]},
            "vase":      {"with_flowers":    ["a vase with flowers", "flowers in a vase"],
                          "without":          ["an empty vase", "a vase without flowers"]},
            "clock":     {"analog": ["an analog clock", "a clock with hands"],
                          "digital":["a digital clock", "a clock with digits"]},
            "umbrella":  {"open":   ["an open umbrella", "umbrella expanded"],
                          "closed": ["a closed umbrella", "umbrella folded"]},
            "laptop":    {"open":   ["an open laptop", "laptop lid open"],
                          "closed": ["a closed laptop", "laptop shut"]},
            "keyboard":  {"typing": ["a hand typing on a keyboard", "keyboard being typed"],
                          "idle":   ["an untouched keyboard", "keyboard not in use"]},
            "kite":      {"flying": ["a kite flying in the sky", "kite airborne"],
                          "grounded":["a kite on the ground", "kite not flying"]},
            "wine_glass":{"filled": ["a filled wine glass", "wine in a glass"],
                          "empty":  ["an empty wine glass", "glass without wine"]},
            "bicycle":   {"ridden": ["a bicycle being ridden", "person on bicycle"],
                          "parked": ["a bicycle parked", "bicycle not in motion"]},
            "airplane":  {"flying": ["an airplane flying in the sky", "an airplane airborne"],
                          "grounded":["an airplane on the ground", "an airplane on a runway"]}
        }

        MATERIAL_TEMPLATES = {
            "table":     {"wooden": ["a wooden table", "a table made of wood"],
                          "plastic":  ["a plastic table", "a table made of plastic"],
                          "metal":    ["a metal table", "a table made of metal"]},
                          "glass":    ["a glass table", "a table made of glass"],
                          "stone":    ["a stone table", "a table made of stone"],
            "chair":     {"wooden": ["a wooden chair", "a chair made of wood"],
                          "plastic":  ["a plastic chair", "a chair made of plastic"],
                          "metal":    ["a metal chair", "a chair made of metal"]},
            "bench":     {"wooden": ["a wooden bench", "a bench made of wood"],
                          "plastic":  ["a plastic bench", "a bench made of plastic"],
                          "metal":    ["a metal bench", "a bench made of metal"]},
            "clock":     {"wooden": ["a wooden clock", "a wall clock made of wood"],
                          "plastic":  ["a plastic clock", "a clock made of plastic"],
                          "metal":    ["a metal clock", "a clock made of metal"]},
            "vase":     {"ceramic": ["a ceramic vase", "a vase made of ceramic"],
                          "glass":  ["a glass vase", "a vase made of glass"],
                          "plastic":  ["a plastic vase", "a vase made of plastic"]},
                          "metal":  ["a metal vase", "a vase made of metal"],
                          "wooden":  ["a wooden vase", "a vase made of wood"],
            "cup":     {"ceramic": ["a ceramic cup", "a mug made of ceramic"],
                          "glass":  ["a glass cup", "a cup made of glass"],
                          "plastic":  ["a plastic cup", "a cup made of plastic"],
                          "paper":  ["a paper cup", "a cup made of paper"]},
                          "metal":  ["a metal cup", "a cup made of metal"],
                          "wooden":  ["a wooden cup", "a cup made of wood"],
            "bowl":     {"ceramic": ["a ceramic bowl", "a bowl made of ceramic"],
                          "glass":  ["a glass bowl", "a bowl made of glass"],
                          "plastic":  ["a plastic bowl", "a bowl made of plastic"],
                          "paper":  ["a paper bowl", "a bowl made of paper"]},
                          "metal":  ["a metal bowl", "a bowl made of metal"],
                          "wooden":  ["a wooden bowl", "a bowl made of wood"]
        }

        templates = STATE_TEMPLATES.get(obj) if self.mode == "state" else MATERIAL_TEMPLATES.get(obj)
        if templates is None:
            # fallback if we don't know this obj
            return ["unknown"] * len(boxes)

        # build prompt → state mapping
        prompt_map = {
            tmpl: state
            for state, tlist in templates.items()
            for tmpl in tlist
        }
        prompts = list(prompt_map.keys())

        # make context crops
        H_img, W_img = image.shape[:2]
        crops: List[Image.Image] = []
        for x1, y1, x2, y2 in boxes:
            w, h = x2 - x1, y2 - y1
            pad_x, pad_y = w * context_margin, h * context_margin
            xi1 = int(max(x1 - pad_x, 0))
            yi1 = int(max(y1 - pad_y, 0))
            xi2 = int(min(x2 + pad_x, W_img))
            yi2 = int(min(y2 + pad_y, H_img))
            crop = image[yi1:yi2, xi1:xi2]
            # SigLIP wants uint8
            if crop.max() <= 1:
                crop = (crop * 255).astype(np.uint8)
            crops.append(Image.fromarray(crop))
            # display(Image.fromarray(crop))

        # run classification
        outputs = self._siglip(
            crops,
            candidate_labels=prompts,
            batch_size=len(crops),
        )

        # aggregate and pick best state per crop
        results: List[str] = []
        for out in outputs:
            scores: dict = {}
            for pred in out:
                state = prompt_map[pred["label"]]
                scores[state] = max(scores.get(state, 0.0), pred["score"])
            # choose highest‐scoring state
            best_state, best_score = max(scores.items(), key=lambda kv: kv[1])
            results.append(best_state if best_score >= threshold else "unknown")

        return results