import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from loguru import logger
from PIL import Image

from deimkit.utils import save_only_ema_weights

from .config import Config
from .visualization import draw_on_image

MODEL_CHECKPOINT_URLS = {
    "deim_hgnetv2_n": "1ZPEhiU9nhW4M5jLnYOFwTSLQC1Ugf62e",  # Nano model
    "deim_hgnetv2_s": "1tB8gVJNrfb6dhFvoHJECKOF5VpkthhfC",  # Small model
    "deim_hgnetv2_m": "18Lj2a6UN6k_n_UzqnJyiaiLGpDzQQit8",  # Medium model
    "deim_hgnetv2_l": "1PIRf02XkrA2xAD3wEiKE2FaamZgSGTAr",  # Large model
    "deim_hgnetv2_x": "1dPtbgtGgq1Oa7k_LgH1GXPelg1IVeu0j",  # XLarge model
}

DEFAULT_CACHE_DIR = os.path.expanduser("./checkpoints")
DEFAULT_IMAGE_SIZE = (640, 640)
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


@dataclass
class PredictionResult:
    """Structured container for prediction results"""

    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    class_names: Optional[List[str]] = None
    visualization: Optional[Image.Image] = None


@dataclass
class ModelConfig:
    """Configuration for model inference"""

    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
    mean: List[float] = field(default_factory=lambda: DEFAULT_MEAN)
    std: List[float] = field(default_factory=lambda: DEFAULT_STD)


class DeployableModel(nn.Module):
    """Wrapper for model deployment with postprocessing"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs


def load_model(
    model_name: str,
    device: str = "auto",
    checkpoint: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    image_size: Optional[Tuple[int, int]] = None,
    ema_weights: bool = True,
) -> "Predictor":
    """Load a DEIM model

    Args:
        model_name: Model name string (one of: 'deim_hgnetv2_n', 'deim_hgnetv2_s',
                   'deim_hgnetv2_m', 'deim_hgnetv2_l', 'deim_hgnetv2_x')
        device: Device to run inference on ('cpu', 'cuda', 'cuda:0', etc. or 'auto')
        checkpoint: Optional path to a custom checkpoint file
        class_names: Optional list of class names for the model
        image_size: Optional custom image size for inference (width, height)
        ema_weights: Whether to load the EMA weights

    Returns:
        Initialized Predictor object
    """
    # if checkpoint:
    #     if ema_weights:
    #         logger.info("Loading EMA weights")
    #         checkpoint = save_only_ema_weights(checkpoint)
    #     else:
    #         logger.info("Loading non-EMA model weights")

    return Predictor(model_name, device, checkpoint, class_names, image_size)


class Predictor:
    """DEIM model predictor for object detection"""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        checkpoint: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ):
        """Initialize a predictor with a DEIM model

        Args:
            model_name: Model name string (one of: 'deim_hgnetv2_n', 'deim_hgnetv2_s',
                       'deim_hgnetv2_m', 'deim_hgnetv2_l', 'deim_hgnetv2_x')
            device: Device to run inference on ('cpu', 'cuda', 'cuda:0', etc. or 'auto')
            checkpoint: Optional path to a custom checkpoint file
            class_names: Optional list of class names for the model
            image_size: Optional custom image size for inference (width, height)
        """
        logger.info(f"Initializing Predictor with device={device}")

        # Validate model name
        self._validate_model_name(model_name)

        # Set up device
        self.device = self._setup_device(device)

        # Set up model configuration
        self.model_config = ModelConfig()
        if image_size is not None:
            logger.info(f"Using custom image size: {image_size}")
            self.model_config.image_size = image_size

        # Load checkpoint
        checkpoint_path = self._get_checkpoint_path(model_name, checkpoint)
        state_dict, num_classes = self._load_checkpoint_state(checkpoint_path)

        # Set up model configuration
        self.cfg = self._setup_model_config(
            model_name, num_classes, self.model_config.image_size
        )

        # Load model weights
        self._load_model_weights(state_dict)

        # Create deployable model
        self.model = DeployableModel(self.cfg).to(self.device)
        self.model.eval()

        # Set up image transforms
        self.transforms = self._create_transforms()

        # Set up class names
        self.class_names = self._setup_class_names(class_names, num_classes)

        logger.success("Predictor initialization complete")

    def _validate_model_name(self, model_name: str) -> None:
        """Validate that the model name is supported"""
        if model_name not in MODEL_CHECKPOINT_URLS:
            raise ValueError(
                f"Invalid model_name: {model_name}. Must be one of {list(MODEL_CHECKPOINT_URLS.keys())}"
            )

    def _setup_device(self, device: str) -> str:
        """Set up and validate the device for inference"""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Auto-selected device: {device}")
        return device

    def _get_checkpoint_path(
        self, model_name: str, custom_checkpoint: Optional[str]
    ) -> str:
        """Get the path to the checkpoint file"""
        if custom_checkpoint is not None:
            if not os.path.exists(custom_checkpoint):
                raise ValueError(f"Custom checkpoint not found: {custom_checkpoint}")
            logger.info(f"Using custom checkpoint: {custom_checkpoint}")
            return custom_checkpoint

        # Download or use cached checkpoint
        return self._download_checkpoint(model_name, MODEL_CHECKPOINT_URLS[model_name])

    def _download_checkpoint(self, model_name: str, file_id: str) -> str:
        """Download checkpoint from Google Drive if not already present"""
        import gdown

        # Create cache directory if it doesn't exist
        cache_dir = Path(DEFAULT_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Construct local path
        local_path = cache_dir / f"{model_name}.pth"

        # Download if not already present
        if not local_path.exists():
            logger.info(f"Downloading checkpoint for model {model_name}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(local_path), quiet=False)
            logger.success(f"Downloaded checkpoint to {local_path}")
        else:
            logger.info(f"Using cached checkpoint from {local_path}")

        return str(local_path)

    def _load_checkpoint_state(self, checkpoint_path: str) -> Tuple[Dict, int]:
        """Load checkpoint and extract state dict and number of classes"""
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Invalid checkpoint_path: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract state dict
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]

        # Determine number of classes from checkpoint
        num_classes = None
        for key, value in state.items():
            if "class_embed" in key and len(value.shape) == 2:
                num_classes = value.shape[0] - 1  # minus "background" class
                break

        return state, num_classes

    def _setup_model_config(
        self, model_name: str, num_classes: Optional[int], image_size: Tuple[int, int]
    ) -> Config:
        """Set up model configuration"""
        logger.info(f"Loading configuration from model name: {model_name}")
        cfg = Config.from_model_name(model_name)

        from deimkit.dataset import _update_image_size

        _update_image_size(cfg, image_size)

        if num_classes is not None:
            logger.info(f"Updating model configuration for {num_classes} classes")
            cfg.yaml_cfg["num_classes"] = num_classes

        # Disable pretrained flag to avoid downloading weights
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        return cfg

    def _load_model_weights(self, state_dict: Dict) -> None:
        """Load weights into the model, adapting parameters if necessary"""
        try:
            self.cfg.model.load_state_dict(state_dict)
            logger.info("Successfully loaded checkpoint weights")
        except RuntimeError as e:
            logger.warning(f"Could not load checkpoint with non-strict loading: {e}")
            logger.warning("Attempting to adapt parameters with shape mismatches...")

            adapted_state_dict = self._adapt_state_dict(state_dict)

            # Load adapted parameters
            self.cfg.model.load_state_dict(adapted_state_dict, strict=False)
            logger.info(
                f"Loaded {len(adapted_state_dict)}/{len(state_dict)} parameters from checkpoint (with adaptations)"
            )

    def _adapt_state_dict(self, state_dict: Dict) -> Dict:
        """Adapt state dict to handle parameter shape mismatches"""
        model_dict = self.cfg.model.state_dict()
        adapted_state_dict = {}

        for k, checkpoint_param in state_dict.items():
            if k not in model_dict:
                continue  # Skip parameters not in model

            model_param = model_dict[k]

            if checkpoint_param.shape == model_param.shape:
                # Shapes match, use directly
                adapted_state_dict[k] = checkpoint_param
            else:
                # Handle shape mismatches based on parameter type
                adapted_state_dict[k] = self._adapt_parameter(
                    k, checkpoint_param, model_param
                )

        return adapted_state_dict

    def _adapt_parameter(
        self, key: str, checkpoint_param: torch.Tensor, model_param: torch.Tensor
    ) -> torch.Tensor:
        """Adapt a single parameter to handle shape mismatches"""
        # For 2D tensors (weights of linear layers)
        if len(checkpoint_param.shape) == 2 and len(model_param.shape) == 2:
            if "class_embed" in key or "score_head" in key:
                # For classification heads, adapt the number of classes
                logger.info(
                    f"Adapting parameter {key}: {checkpoint_param.shape} -> {model_param.shape}"
                )

                # Initialize with the model's random weights
                adapted_param = model_param.clone()

                # Copy weights for common classes
                min_classes = min(checkpoint_param.shape[0], model_param.shape[0])
                min_features = min(checkpoint_param.shape[1], model_param.shape[1])
                adapted_param[:min_classes, :min_features] = checkpoint_param[
                    :min_classes, :min_features
                ]

                return adapted_param
            else:
                # For other 2D tensors, try to adapt if possible
                logger.info(
                    f"Adapting parameter {key}: {checkpoint_param.shape} -> {model_param.shape}"
                )
                adapted_param = model_param.clone()
                min_dim0 = min(checkpoint_param.shape[0], model_param.shape[0])
                min_dim1 = min(checkpoint_param.shape[1], model_param.shape[1])
                adapted_param[:min_dim0, :min_dim1] = checkpoint_param[
                    :min_dim0, :min_dim1
                ]
                return adapted_param

        # For 1D tensors (biases, etc.)
        elif len(checkpoint_param.shape) == 1 and len(model_param.shape) == 1:
            logger.info(
                f"Adapting parameter {key}: {checkpoint_param.shape} -> {model_param.shape}"
            )
            adapted_param = model_param.clone()
            min_dim = min(checkpoint_param.shape[0], model_param.shape[0])
            adapted_param[:min_dim] = checkpoint_param[:min_dim]
            return adapted_param

        # For other tensor shapes, use model's initialization
        else:
            logger.warning(
                f"Complex shape mismatch for {key}: {checkpoint_param.shape} vs {model_param.shape}. Using model's initialization."
            )
            return model_param

    def _create_transforms(self) -> T.Compose:
        """Create image transforms for preprocessing"""
        return T.Compose(
            [
                T.Resize(self.model_config.image_size),
                T.ToTensor(),
                T.Normalize(mean=self.model_config.mean, std=self.model_config.std),
            ]
        )

    def _setup_class_names(
        self, class_names: Optional[List[str]], num_classes: Optional[int]
    ) -> List[str]:
        """Set up class names for the model"""
        if class_names is not None:
            return class_names

        # Create default class names if none provided
        if num_classes is None:
            num_classes = self.cfg.yaml_cfg["num_classes"]

        default_names = [f"Class_{i}" for i in range(num_classes)]
        logger.debug(
            f"No class names provided. Created {num_classes} default class names."
        )
        return default_names

    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        visualize: bool = False,
        save_path: Optional[str] = None,
    ) -> PredictionResult:
        """Run inference on a single image path

        Args:
            image_path: Path to the image file
            conf_threshold: Confidence threshold for detections
            visualize: Whether to return visualization image
            save_path: Optional path to save the visualization

        Returns:
            PredictionResult object containing detection results
        """
        try:
            # Load and validate image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            logger.debug(f"Loading image from path: {image_path}")
            im_pil = Image.open(image_path).convert("RGB")

            # Process image and run inference
            return self._process_single_image(
                im_pil,
                conf_threshold=conf_threshold,
                visualize=visualize,
                save_path=save_path,
            )

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise RuntimeError(f"Error processing image: {str(e)}")

    def _process_single_image(
        self,
        image: Image.Image,
        conf_threshold: float = 0.25,
        visualize: bool = False,
        save_path: Optional[str] = None,
    ) -> PredictionResult:
        """Process a single PIL image and return predictions"""
        # Store original image for visualization
        original_image = image.copy() if visualize else None

        # Get original dimensions
        w, h = image.size

        # Preprocess image
        im_data = self.transforms(image)
        im_data = im_data.unsqueeze(0).to(self.device)
        orig_sizes = torch.tensor([[w, h]], device=self.device)

        # Run inference
        labels, boxes, scores = self.model(im_data, orig_sizes)

        # Get results for first (and only) image
        labels = labels[0]
        boxes = boxes[0]
        scores = scores[0]

        # Filter by confidence
        mask = scores > conf_threshold
        filtered_boxes = boxes[mask].cpu().numpy()
        filtered_labels = labels[mask].cpu().numpy()
        filtered_scores = scores[mask].cpu().numpy()

        # Map numeric labels to class names
        class_names = None
        if self.class_names is not None:
            class_names = [
                self.class_names[int(label)]
                if 0 <= int(label) < len(self.class_names)
                else f"unknown_{int(label)}"
                for label in filtered_labels
            ]

        logger.debug(f"Prediction complete. Found {len(filtered_boxes)} objects")

        # Create result object
        result = PredictionResult(
            boxes=filtered_boxes,
            labels=filtered_labels,
            scores=filtered_scores,
            class_names=class_names,
        )

        # Add visualization if requested
        if visualize:
            logger.debug("Generating visualization")
            vis_image = draw_on_image(
                image=original_image,
                detections={
                    "boxes": filtered_boxes,
                    "labels": filtered_labels,
                    "scores": filtered_scores,
                    "class_names": class_names,
                },
                score_threshold=conf_threshold,
                output_path=save_path,
                class_names=self.class_names,
            )
            result.visualization = vis_image

        return result

    @torch.no_grad()
    def predict_batch(
        self,
        image_paths: List[str],
        conf_threshold: float = 0.25,
        visualize: bool = False,
        batch_size: int = 16,
        save_path: Optional[str] = None,
    ) -> List[PredictionResult]:
        """Run inference on a batch of image paths efficiently

        Args:
            image_paths: List of paths to image files
            conf_threshold: Confidence threshold for detections
            visualize: Whether to return visualization images
            batch_size: Batch size for processing (defaults to 16)
            save_path: Optional path prefix to save visualizations

        Returns:
            List of PredictionResult objects
        """
        logger.info(
            f"Processing batch of {len(image_paths)} images with batch_size={batch_size}"
        )
        results = []

        # Process images in batches
        for batch_idx in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[batch_idx : batch_idx + batch_size]
            batch_results = self._process_image_batch(
                batch_paths, batch_idx, conf_threshold, visualize, save_path
            )
            results.extend(batch_results)

        logger.success(f"Batch processing complete. Processed {len(results)} images")
        return results

    def _process_image_batch(
        self,
        batch_paths: List[str],
        batch_start_idx: int,
        conf_threshold: float,
        visualize: bool,
        save_path: Optional[str],
    ) -> List[PredictionResult]:
        """Process a batch of images and return predictions"""
        batch = []
        orig_sizes = []
        original_images = []
        batch_results = []

        # Load and preprocess images
        for i, image_path in enumerate(batch_paths):
            try:
                # Load image
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")

                logger.debug(
                    f"Loading image {batch_start_idx + i} from path: {image_path}"
                )
                im_pil = Image.open(image_path).convert("RGB")

                # Store original image for visualization if needed
                if visualize:
                    original_images.append(im_pil.copy())

                # Get original dimensions
                w, h = im_pil.size
                orig_sizes.append([w, h])

                # Preprocess image
                im_data = self.transforms(im_pil)
                batch.append(im_data)

            except Exception as e:
                logger.error(
                    f"Error loading image at index {batch_start_idx + i}: {str(e)}"
                )
                # Add placeholder for failed image
                batch_results.append(
                    PredictionResult(
                        boxes=np.array([]),
                        labels=np.array([]),
                        scores=np.array([]),
                    )
                )

        # If we have images to process
        if batch:
            try:
                # Run inference on batch
                batch_tensor = torch.stack(batch).to(self.device)
                orig_sizes_tensor = torch.tensor(orig_sizes).to(self.device)

                # Run inference
                labels, boxes, scores = self.model(batch_tensor, orig_sizes_tensor)

                # Process each image result
                for i in range(len(batch)):
                    mask = scores[i] > conf_threshold
                    filtered_labels = labels[i][mask].cpu().numpy()
                    filtered_boxes = boxes[i][mask].cpu().numpy()
                    filtered_scores = scores[i][mask].cpu().numpy()

                    # Map numeric labels to class names
                    class_names = None
                    if self.class_names is not None:
                        class_names = [
                            self.class_names[int(label)]
                            if 0 <= int(label) < len(self.class_names)
                            else f"unknown_{int(label)}"
                            for label in filtered_labels
                        ]

                    # Create result object
                    result = PredictionResult(
                        boxes=filtered_boxes,
                        labels=filtered_labels,
                        scores=filtered_scores,
                        class_names=class_names,
                    )

                    # Add visualization if requested
                    if visualize:
                        logger.debug(
                            f"Generating visualization for image {batch_start_idx + i}"
                        )
                        output_path = (
                            f"{save_path}_{batch_start_idx + i}.png"
                            if save_path
                            else None
                        )
                        vis_image = draw_on_image(
                            image=original_images[i],
                            detections={
                                "boxes": filtered_boxes,
                                "labels": filtered_labels,
                                "scores": filtered_scores,
                                "class_names": class_names,
                            },
                            score_threshold=conf_threshold,
                            output_path=output_path,
                            class_names=self.class_names,
                        )
                        result.visualization = vis_image

                    batch_results.append(result)

            except Exception as e:
                logger.error(
                    f"Error processing batch starting at index {batch_start_idx}: {str(e)}"
                )
                # Add placeholders for all images in failed batch
                for _ in range(len(batch)):
                    batch_results.append(
                        PredictionResult(
                            boxes=np.array([]),
                            labels=np.array([]),
                            scores=np.array([]),
                        )
                    )

        return batch_results
