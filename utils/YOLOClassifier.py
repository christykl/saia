import os
import warnings
from pathlib import Path
from typing import Sequence, Tuple

import requests
import torch
from PIL import Image
from tqdm.auto import tqdm
from ultralytics import YOLO

warnings.filterwarnings('ignore')
_current_dir = Path(__file__).resolve().parent


class YOLOClassifier:
    """A class"""

    def __init__(
        self,
        labels: Sequence[str],
        mode: str,
        bias: str,
        bias_discount: float = 0,
        device: str = 'cpu',
    ) -> None:
        """
        Initialize YOLOClassifier with specified parameters.

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

        # Download and Load Model
        self.model = YOLO(self.download_model()).to(self.device)

    def download_model(self):
        """Download YOLO weights if needed"""
        out_path = os.path.join(_current_dir, 'YOLO', 'yolo.pt')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if not os.path.exists(out_path):
            url = (
                'https://www.comet.com/api/asset/download?assetId='
                'c8879209accf4f5da69cae4026c9c9cd&experimentKey=626748230dcb4cfb96bbb775dc76edc0'
            )

            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total = int(r.headers.get('Content-Length', 0))  # bytes

                with (
                    open(out_path, 'wb') as f,
                    tqdm(
                        total=total,
                        unit='iB',  # binary bytes
                        unit_scale=True,  # auto-scale to KiB/MiB/GiB
                        unit_divisor=1024,  # 1024-based
                        desc='Downloading YOLO...',
                    ) as pbar,
                ):
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        return out_path

    def calc_score(self, image: Image.Image) -> Tuple[float, Image.Image, str]:
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
        det = self.model.predict(image, imgsz=(352, 512), verbose=False)[0].summary()
        logit = max(
            [0.0] + [data['confidence'] for data in det if data['name'] in self.labels]
        )

        image = image.resize((224, 224), Image.LANCZOS)
        tag = 'full' if logit > 0.5 else 'discounted'

        return round(logit, 2), image
