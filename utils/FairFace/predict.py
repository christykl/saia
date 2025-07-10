from __future__ import print_function, division
from torchvision.transforms.functional import resize
import torch
import numpy as np
import os
import dlib
from typing import *
from torchvision import models, transforms
from uniface import RetinaFace, face_alignment
from uniface.constants import RetinaFaceWeights
from torch import nn
from PIL import Image


class FaceAnalyzer:
    """Main class for face detection and attribute analysis"""

    # Mapping dictionaries for classiification results
    RACE_MAP = {
        0: "white",
        1: "black",
        2: "latino_hispanic",
        3: "east_asian",
        4: "southeast_asian",
        5: "indian",
        6: "middle_eastern",
    }

    GENDER_MAP = {0: "male", 1: "female"}

    AGE_MAP = {
        0: "young",  # 0-2
        1: "young",  # 3-9
        2: "young",  # 10-19
        3: "young",  # 20-29
        4: "middle-aged",  # 30-39
        5: "middle-aged",  # 40-49
        6: "old",  # 50-59
        7: "old",  # 60-69
        8: "old",  # 70+
    }

    def __init__(
        self,
        model_base_path: str,
        detector: str = "dlib",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize face detector and classifier models

        Args:
            model_base_path: Path to model directory
            detector: Which face detector to use ('dlib' or 'retinaface')
            device: Device to run models on ('cuda' or 'cpu')
        """
        if detector not in ["dlib", "retinaface"]:
            raise ValueError("detector must be 'dlib' or 'retinaface'")

        self.device = torch.device(device)
        self.model_base_path = model_base_path
        self.detector_type = detector  # store it for later use

        if detector == "dlib":
            # Initialize dlib models
            self.dlib_model_path = os.path.join(model_base_path, "dlib_models")
            self.cnn_face_detector = dlib.cnn_face_detection_model_v1(
                os.path.join(self.dlib_model_path, "mmod_human_face_detector.dat")
            )
            self.shape_predictor = dlib.shape_predictor(
                os.path.join(
                    self.dlib_model_path, "shape_predictor_5_face_landmarks.dat"
                )
            )
        else:
            # Initialize RetinaFace
            self.retinaface = RetinaFace(
                model_name=RetinaFaceWeights.MNET_V2,
                conf_thresh=0.5,
                pre_nms_topk=5000,
                nms_thresh=0.4,
                post_nms_topk=750,
                dynamic_size=True,
            )

        # Initialize FairFace model (always needed)
        self.fairface_model_path = os.path.join(model_base_path, "fair_face_models")
        self.fairface = models.resnet34(pretrained=True)
        self.fairface.fc = nn.Linear(self.fairface.fc.in_features, 18)
        self.fairface.load_state_dict(
            torch.load(
                os.path.join(self.fairface_model_path, "fairface_alldata_20191111.pt")
            )
        )
        self.fairface = self.fairface.to(device)
        self.fairface.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    # Initialize dlib models
    def detect_face_dlib(
        self, image: np.ndarray, default_max_size=800, size=300, padding=0.25
    ) -> List[np.ndarray]:
        """Detect faces using dlib's CNN face detector

        Args:
            image: Input image as numpy array
            default_max_size: Maximum dimension for resizing
            size: Output face image size
            padding: Padding around detected face

        Returns:
            List of detected face images
        """
        # Resize image for better performance
        old_height, old_width, _ = image.shape
        if old_width > old_height:
            new_width, new_height = (
                default_max_size,
                int(default_max_size * old_height / old_width),
            )
        else:
            new_width, new_height = (
                int(default_max_size * old_width / old_height),
                default_max_size,
            )
        image = dlib.resize_image(image, rows=new_height, cols=new_width)

        # Detect faces
        dets = self.cnn_face_detector(image, 1)
        num_faces = len(dets)
        if num_faces == 0:
            return []

        # Get face landmarks
        faces = dlib.full_object_detections()
        for detection in dets:
            rect = detection.rect
            faces.append(self.shape_predictor(image, rect))

        # Crop faces
        return dlib.get_face_chips(image, faces, size=size, padding=padding)

    def _expand_bbox(self, bbox, padding_factor, image_width, image_height):
        x1, y1, x2, y2, c = bbox
        width = x2 - x1
        height = y2 - y1
        center_x = x1 + width / 2
        center_y = y1 + height / 2

        new_width = width * padding_factor
        new_height = height * padding_factor

        new_x1 = max(int(center_x - new_width / 2), 0)
        new_y1 = max(int(center_y - new_height / 2), 0)
        new_x2 = min(int(center_x + new_width / 2), image_width - 1)
        new_y2 = min(int(center_y + new_height / 2), image_height - 1)

        return [new_x1, new_y1, new_x2, new_y2, c]

    def _adjust_landmarks(self, landmarks, original_bbox, expanded_bbox):
        x1_orig, y1_orig, _, _, _ = original_bbox
        x1_exp, y1_exp, _, _, _ = expanded_bbox
        dx = x1_orig - x1_exp
        dy = y1_orig - y1_exp

        adjusted_landmarks = []
        for x, y in landmarks:
            adjusted_landmarks.append((x - dx, y - dy))
        return adjusted_landmarks

    def _add_padding(self, img, bbox, landmarks, padding_factor):
        new_bbox = self._expand_bbox(bbox, padding_factor, img.shape[1], img.shape[0])
        new_landmarks = self._adjust_landmarks(landmarks, bbox, new_bbox)
        return new_bbox, new_landmarks

    def detect_face_retina(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect faces using RetinaFace

        Args:
            image: Input image as numpy array
            size: Input size for RetinaFace model

        Returns:
            List containing detected face image
        """

        # RetinaFace expects BGR
        image_bgr = image[..., ::-1]

        # Detect faces
        detections, landmarks = self.retinaface.detect(image_bgr)
        for i in range(len(detections)):
            old_bbox = detections[i]
            detections[i] = self._expand_bbox(
                detections[i], 0.25, image_bgr.shape[1], image_bgr.shape[0]
            )
            landmarks[i] = self._adjust_landmarks(landmarks[i], old_bbox, detections[i])

        if not detections.size:
            return []

        # Align faces
        cropped_images = [
            face_alignment(image_bgr, l, image_size=512)[0] for l in landmarks
        ]

        cropped_images = [
            img[..., ::-1] for img in cropped_images
        ]  # Convert back to RGB

        return cropped_images

    def predict_attributes(
        self, face_images: List[np.ndarray]
    ) -> Tuple[List[int], List[int], List[int]]:
        """Predict age, gender, and race for detected faces

        Args:
            face_images: List of face images as numpy arrays

        Returns:
            Tuple of (age_predictions, gender_predictions, race_predictions)
        """
        race_pred = []
        gender_pred = []
        age_pred = []

        with torch.no_grad():
            for face_img in face_images:
                # Convert to PIL and apply transforms
                face_pil = Image.fromarray(face_img)
                face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)

                # Get model predictions
                outputs = self.fairface(face_tensor).cpu().detach().numpy().squeeze()

                # Split outputs
                race_outputs = outputs[:7]
                gender_outputs = outputs[7:9]
                age_outputs = outputs[9:18]

                # Apply softmax and get class predictions
                race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
                gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
                age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

                race_pred.append(np.argmax(race_score))
                gender_pred.append(np.argmax(gender_score))
                age_pred.append(np.argmax(age_score))

        return age_pred, gender_pred, race_pred

    def analyze_image(
        self, image: np.ndarray, mode: str = "all", detector: str = "dlib"
    ) -> Dict:
        """Analyze an image for face attributes

        Args:
            image: Input image as numpy array
            mode: Analysis mode ('age', 'gender', 'race', or 'all')
            detector: Face detector to use ('dlib' or 'retinaface')

        Returns:
            Dictionary with analysis results
        """
        # Validate inputs
        if mode not in ["age", "gender", "race", "all"]:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'age', 'gender', 'race', or 'all'"
            )

        if detector not in ["dlib", "retinaface"]:
            raise ValueError(
                f"Invalid detector: {detector}. Must be 'dlib' or 'retinaface'"
            )

        # Detect faces
        if detector == "dlib":
            face_images = self.detect_face_dlib(image)
        else:
            face_images = self.detect_face_retina(image)

        if not face_images:
            return {"faces_detected": 0, "predictions": {}}

        # Predict attributes
        age_preds, gender_preds, race_preds = self.predict_attributes(face_images)

        # Map to human-readable labels
        age_labels = [self.AGE_MAP[age] for age in age_preds]
        gender_labels = [self.GENDER_MAP[gender] for gender in gender_preds]
        race_labels = [self.RACE_MAP[race] for race in race_preds]

        # Format results based on mode
        results = {"faces_detected": len(face_images), "predictions": {}}

        if mode == "all":
            results["predictions"] = {
                "age": age_labels,
                "gender": gender_labels,
                "race": race_labels,
            }
        elif mode == "age":
            results["predictions"] = {"age": age_labels}
        elif mode == "gender":
            results["predictions"] = {"gender": gender_labels}
        else:  # race
            results["predictions"] = {"race": race_labels}

        return results


def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
