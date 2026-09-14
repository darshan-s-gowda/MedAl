"""
CNN Model for Medical Image Analysis (Chest X-Ray Classification)
Classifies: Normal vs Pneumonia (extensible to other conditions)

Architecture: Custom lightweight CNN + optional transfer learning with ResNet18
Uses: PyTorch
"""

import os
import logging
import io
import base64
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Attempt to import torch - graceful fallback if not installed
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms, models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. CNN image analysis will be unavailable.")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# CNN Architecture
# ─────────────────────────────────────────────

if TORCH_AVAILABLE:

    class MedicalCNN(nn.Module):
        """
        Lightweight CNN for chest X-ray classification.
        Input: 224x224 grayscale image (converted to 3-channel for compatibility)
        Output: 2 classes (Normal, Pneumonia)
        
        Architecture inspired by VGG-style but much lighter for fast inference.
        """

        def __init__(self, num_classes: int = 2):
            super(MedicalCNN, self).__init__()

            # Feature extraction blocks
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # 112x112
                nn.Dropout2d(0.1),

                # Block 2
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # 56x56
                nn.Dropout2d(0.2),

                # Block 3
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # 28x28
                nn.Dropout2d(0.25),

                # Block 4
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # 14x14
            )

            # Global Average Pooling (replaces large FC layer)
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

            # Classifier head
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = self.global_avg_pool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x


# ─────────────────────────────────────────────
# Image Preprocessing
# ─────────────────────────────────────────────

def get_transform() -> Any:
    """Standard preprocessing for chest X-ray images."""
    if not TORCH_AVAILABLE:
        return None
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # Convert to 3ch
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means (works well for X-rays too)
            std=[0.229, 0.224, 0.225]
        )
    ])


# ─────────────────────────────────────────────
# X-Ray Analyzer (API-facing)
# ─────────────────────────────────────────────

class XRayAnalyzer:
    """
    Analyzes chest X-ray images using the MedicalCNN architecture defined above.

    This is the SINGLE, canonical X-ray inference pipeline for the project
    (the older ResNet18/torch.hub-based implementation has been removed —
    see README "X-Ray Model" section for details).

    IMPORTANT — Model status:
    Real trained weights are NOT bundled with this project (training on a
    real chest X-ray dataset, e.g. NIH ChestX-ray14 or the Kaggle
    "Chest X-Ray Images (Pneumonia)" dataset, requires downloading a large
    labeled dataset which is outside the scope of what ships in this repo).
    See `src/ml/train_xray.py` for the real training pipeline — point it at
    a dataset on disk and it will produce `models/xray_cnn.pth`.

    Until that file exists, this analyzer clearly reports itself as
    UNTRAINED and does not return a pneumonia/normal classification, so the
    API never implies a working diagnostic model where none exists.

    Usage:
        analyzer = XRayAnalyzer()
        result = analyzer.analyze_image(base64_string)
    """

    CLASSES = ["Normal", "Pneumonia"]
    WEIGHTS_PATH = MODEL_DIR / "xray_cnn.pth"
    NON_CLINICAL_DISCLAIMER = (
        "AI-assisted screening aid only — NOT a medical diagnosis. "
        "Always confirm findings with a qualified radiologist/physician."
    )

    def __init__(self):
        self.model = None
        self.trained = False
        self.device = None

        if not TORCH_AVAILABLE:
            logger.warning("XRayAnalyzer disabled: PyTorch not available.")
            return

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.transform = get_transform()
            self.model, self.trained = self._load_model()
            logger.info(
                f"XRayAnalyzer initialized on {self.device} "
                f"(trained_weights_loaded={self.trained})"
            )
        except Exception as e:
            # Never let a bad checkpoint / environment issue take Flask down.
            logger.error(f"Failed to initialize XRayAnalyzer: {e}", exc_info=True)
            self.model = None
            self.trained = False

    def _load_model(self):
        """
        Load trained model weights from models/xray_cnn.pth if present.

        Returns:
            (model, trained: bool) — `trained` is False when no weights file
            exists yet or the checkpoint failed to load, in which case the
            model has freshly-initialized (random) weights and must NOT be
            used to produce a real classification.
        """
        model = MedicalCNN(num_classes=2).to(self.device)

        if self.WEIGHTS_PATH.exists():
            try:
                state = torch.load(self.WEIGHTS_PATH, map_location=self.device,weights_only=True)
                model.load_state_dict(state)
                model.eval()
                logger.info("Loaded trained CNN weights from disk.")
                return model, True
            except Exception as e:
                logger.error(
                    f"Found {self.WEIGHTS_PATH} but failed to load it "
                    f"(corrupted or incompatible checkpoint?): {e}. "
                    "Falling back to untrained mode.",
                    exc_info=True
                )
                model = MedicalCNN(num_classes=2).to(self.device)

        logger.warning(
            f"No trained weights found at {self.WEIGHTS_PATH}. "
            "Running in DEMO / untrained mode — X-ray endpoint will report "
            "itself as unavailable for real predictions until "
            "src/ml/train_xray.py has been run on a real dataset."
        )
        model.eval()
        return model, False

    def analyze_image(self, image_data: str) -> Dict[str, Any]:
        """
        Analyze a chest X-ray image.

        Args:
            image_data: Base64-encoded image string (optionally with a
                `data:image/...;base64,` prefix).

        Returns:
            Dict describing the outcome. Always includes `available` (bool).
            When the model isn't trained yet, `available` is False and no
            prediction is returned, so the frontend never displays a
            confident-looking but meaningless result.
        """
        if not TORCH_AVAILABLE or self.model is None:
            return {
                "available": False,
                "code": "model_unavailable",
                "error": "X-ray model unavailable (PyTorch not installed or failed to load)."
            }

        if not self.trained:
            return {
                "available": False,
                "trained": False,
                "code": "not_trained",
                "error": (
                    "The X-ray classification model has not been trained yet. "
                    "This is a demonstration project without bundled trained "
                    "weights, so no real prediction can be produced. "
                    "See README for how to train the model on a real dataset."
                )
            }

        # ── Decode & validate the image ──
        # (Client input problems get code="invalid_input" so the API layer
        # can return 400 for these, vs. 200 for legitimate "not available"
        # service states above.)
        try:
            payload = image_data
            if "," in payload:
                payload = payload.split(",", 1)[1]
            img_bytes = base64.b64decode(payload, validate=False)
        except Exception:
            return {
                "available": False,
                "code": "invalid_input",
                "error": "Invalid image encoding (expected base64)."
            }

        try:
            img = Image.open(io.BytesIO(img_bytes))
            img.verify()  # Detect corrupted/truncated files early
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # re-open after verify()
        except Exception:
            return {
                "available": False,
                "code": "invalid_input",
                "error": "Uploaded file is not a valid or supported image."
            }

        # ── Preprocess + run inference ──
        try:
            tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1)[0]

            probs_list = probs.cpu().numpy().tolist()
            pred_idx = int(np.argmax(probs_list))
            confidence = probs_list[pred_idx]

            return {
                "available": True,
                "trained": True,
                "prediction": self.CLASSES[pred_idx],
                "confidence": round(confidence * 100, 1),
                "class_probabilities": {
                    cls: round(prob * 100, 1)
                    for cls, prob in zip(self.CLASSES, probs_list)
                },
                "disclaimer": self.NON_CLINICAL_DISCLAIMER
            }

        except Exception as e:
            logger.error(f"X-ray inference error: {e}", exc_info=True)
            return {
                "available": False,
                "code": "inference_error",
                "error": "Image analysis failed while running the model."
            }


if __name__ == "__main__":
    # Quick manual smoke-check: confirms the architecture builds and a
    # forward pass runs. Real training lives in src/ml/train_xray.py.
    logging.basicConfig(level=logging.INFO)
    if TORCH_AVAILABLE:
        m = MedicalCNN(num_classes=2)
        out = m(torch.randn(1, 3, 224, 224))
        logger.info(f"MedicalCNN forward-pass OK, output shape={tuple(out.shape)}")
    else:
        print("PyTorch not available.")
