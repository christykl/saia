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
_current_dir = Path(__file__).resolve().parent


class CLIPClassifier:
    """A class"""

    def __init__(
        self,
        labels: Sequence[str],
        mode: str,
        bias: str,
        bias_discount: float = 0,
        device: str = "cpu",
    ) -> None:
        """
        Initialize CLIPClassifier with specified parameters.

        Args:
            labels: Sequence of object labels to detect
            mode: Analysis mode ('age', 'gender', 'race', 'color', 'pose', 'setting')
            bias: The specific bias to check for
            bias_discount: Discount factor when bias isn't present
            device: Computation device ('cpu' or 'cuda')
        """
        # ----- Classifier Configuration -----
        self.labels = labels
        self.mode = mode.lower()
        self.bias = bias.lower()
        self.bias_discount = 1 - float(bias_discount)
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)


        # Load CLIP model & processor once
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    
    def clip_image_text_similarity(self, image: Image.Image, text_prompt: str) -> float:
        # Image embedding
        image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_emb = self.model.get_image_features(**image_inputs)

        # Text embedding
        text_inputs = self.processor(text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_emb = self.model.get_text_features(**text_inputs)

        # Normalize to unit vectors
        image_emb = torch.nn.functional.normalize(image_emb, dim=-1)
        text_emb  = torch.nn.functional.normalize(text_emb, dim=-1)

        # Cosine similarity (dot product)
        cosine_sim = (image_emb @ text_emb.T).item()

        return cosine_sim
    
    def calc_score(
        self, image: Image.Image
    ) -> Tuple[float, Image.Image, str]:
        """
        Calculate score values for a single image.

        Args:
            image: PIL image to process

        Returns:
            Tuple containing:
            - Score value
            - Resized image
            - Tag indicating score type (none, full, discounted)
        """
        tag = None

        logit = self.clip_image_text_similarity(image, f"a photo of a {self.labels}")
        image = image.resize((224, 224), Image.LANCZOS)
        tag = "full" if logit > 0.5 else "discounted"
            
        return round(logit, 2), image