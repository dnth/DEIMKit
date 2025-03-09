import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from loguru import logger
from PIL import Image

from .config import Config
from .visualization import draw_on_image


class Predictor:
    def __init__(self, model_name: str, device: str = "auto"):
        """Initialize a predictor with a DEIM model

        Args:
            model_name: Model name string (one of: 'deim_hgnetv2_n', 'deim_hgnetv2_s',
                       'deim_hgnetv2_m', 'deim_hgnetv2_l', 'deim_hgnetv2_x')
            device: Device to run inference on ('cpu', 'cuda', 'cuda:0', etc. or 'auto')
        """
        logger.info(f"Initializing Predictor with device={device}")

        # Model checkpoint URLs
        CHECKPOINT_URLS = {
            "deim_hgnetv2_n": "1ZPEhiU9nhW4M5jLnYOFwTSLQC1Ugf62e",  # Nano model
            "deim_hgnetv2_s": "1tB8gVJNrfb6dhFvoHJECKOF5VpkthhfC",  # Small model
            "deim_hgnetv2_m": "18Lj2a6UN6k_n_UzqnJyiaiLGpDzQQit8",  # Medium model
            "deim_hgnetv2_l": "1PIRf02XkrA2xAD3wEiKE2FaamZgSGTAr",  # Large model
            "deim_hgnetv2_x": "1dPtbgtGgq1Oa7k_LgH1GXPelg1IVeu0j",  # XLarge model
        }

        if model_name not in CHECKPOINT_URLS:
            raise ValueError(
                f"Invalid model_name: {model_name}. Must be one of {list(CHECKPOINT_URLS.keys())}"
            )

        # Auto-select device if specified
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Auto-selected device: {device}")
        self.device = device

        # Download checkpoint if needed
        checkpoint_path = self._download_checkpoint(
            model_name, CHECKPOINT_URLS[model_name]
        )

        # Initialize distributed environment for single-process inference
        # This prevents errors when loading models that check for distributed rank
        if not torch.distributed.is_initialized():
            try:
                # Use file:// method with a temporary file instead of env://
                with tempfile.NamedTemporaryFile(delete=False) as tf:
                    init_file = tf.name
                torch.distributed.init_process_group(
                    backend="gloo",
                    init_method=f"file://{init_file}",
                    rank=0,
                    world_size=1,
                )
                # Clean up the temp file
                if os.path.exists(init_file):
                    os.unlink(init_file)
                logger.debug("Successfully initialized distributed environment")
            except Exception as e:
                logger.warning(f"Failed to initialize distributed environment: {e}")
                logger.info("Falling back to rank function patch")
                # If distributed initialization fails, we'll patch the rank function
                # to avoid the distributed check in the model
                if not hasattr(torch.distributed, "_original_get_rank"):
                    torch.distributed._original_get_rank = torch.distributed.get_rank
                    torch.distributed.get_rank = lambda *args, **kwargs: 0

        # Load configuration
        logger.info(f"Loading configuration from model name: {model_name}")
        self.cfg = Config.from_model_name(model_name)

        # Disable pretrained flag to avoid downloading weights
        if "HGNetv2" in self.cfg.yaml_cfg:
            self.cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        # Load checkpoint
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            raise ValueError(f"Invalid checkpoint_path: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]

        # Load model state
        self.cfg.model.load_state_dict(state)

        # Create model for inference
        class Model(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()

            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs

        self.model = Model(self.cfg).to(self.device)
        self.model.eval()

        # Enhanced image transforms with normalization
        self.transforms = T.Compose(
            [
                T.Resize((640, 640)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        logger.success("Predictor initialization complete")

    def _download_checkpoint(self, model_name, file_id):
        """Download checkpoint from Google Drive if not already present"""
        import gdown
        import os

        # Create cache directory if it doesn't exist
        cache_dir = os.path.expanduser("~/.cache/deim/checkpoints")
        os.makedirs(cache_dir, exist_ok=True)

        # Construct local path
        local_path = os.path.join(cache_dir, f"{model_name}.pth")

        # Download if not already present
        if not os.path.exists(local_path):
            logger.info(f"Downloading checkpoint for model {model_name}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, local_path, quiet=False)
            logger.success(f"Downloaded checkpoint to {local_path}")
        else:
            logger.info(f"Using cached checkpoint from {local_path}")

        return local_path

    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        visualize: bool = False,
        save_path: str | None = None,
    ):
        """Run inference on a single image path

        Args:
            image_path: Path to the image file
            conf_threshold: Confidence threshold for detections
            visualize: Whether to return visualization image
            save_path: Optional path to save the visualization
        """
        try:
            # Load image from path
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            logger.debug(f"Loading image from path: {image_path}")
            im_pil = Image.open(image_path).convert("RGB")

            # Store original image for visualization
            original_image = im_pil.copy() if visualize else None

            # Get original dimensions
            w, h = im_pil.size

            # Preprocess image
            im_data = self.transforms(im_pil)
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

            results = {
                "boxes": filtered_boxes,
                "labels": filtered_labels,
                "scores": filtered_scores,
            }

            logger.debug(f"Prediction complete. Found {len(filtered_boxes)} objects")

            if visualize:
                logger.debug("Generating visualization")
                vis_image = draw_on_image(
                    image=original_image,
                    detections=results,
                    score_threshold=conf_threshold,
                    output_path=save_path,
                )
                results["visualization"] = vis_image

            return results

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise RuntimeError(f"Error processing image: {str(e)}")

    @torch.no_grad()
    def predict_batch(
        self,
        image_paths: list[str],
        conf_threshold: float = 0.25,
        visualize: bool = False,
        batch_size: int = 16,
        save_path: str | None = None,
    ):
        """Run inference on a batch of image paths efficiently

        Args:
            image_paths: List of paths to image files
            conf_threshold: Confidence threshold for detections
            visualize: Whether to return visualization image
            batch_size: Batch size for processing (defaults to 16)
            save_path: Optional path to save the visualization
        """

        logger.info(
            f"Processing batch of {len(image_paths)} images with batch_size={batch_size}"
        )
        results = []
        batch = []
        orig_sizes = []

        for idx, image_path in enumerate(image_paths):
            try:
                # Load image from path
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                logger.debug(f"Loading image {idx} from path: {image_path}")
                im_pil = Image.open(image_path).convert("RGB")

                # Store original image for visualization if needed
                original_image = im_pil.copy() if visualize else None

                # Get original dimensions
                w, h = im_pil.size
                orig_sizes.append([w, h])

                # Preprocess image
                im_data = self.transforms(im_pil)
                batch.append(im_data)

                # Process batch when full or at end
                if len(batch) == batch_size or idx == len(image_paths) - 1:
                    logger.debug(f"Processing batch of {len(batch)} images")
                    batch_tensor = torch.stack(batch).to(self.device)
                    orig_sizes_tensor = torch.tensor(orig_sizes).to(self.device)

                    # Run inference
                    labels, boxes, scores = self.model(batch_tensor, orig_sizes_tensor)

                    # Process each image in batch
                    for i in range(len(batch)):
                        mask = scores[i] > conf_threshold
                        result = {
                            "boxes": boxes[i][mask].cpu().numpy(),
                            "labels": labels[i][mask].cpu().numpy(),
                            "scores": scores[i][mask].cpu().numpy(),
                        }

                        # Add visualization if requested
                        if visualize:
                            logger.debug(
                                f"Generating visualization for image {idx - len(batch) + 1 + i}"
                            )
                            output_path = (
                                f"{save_path}_{idx - len(batch) + 1 + i}.png"
                                if save_path
                                else None
                            )
                            vis_image = draw_on_image(
                                image=original_image,
                                detections=result,
                                score_threshold=conf_threshold,
                                output_path=output_path,
                            )
                            result["visualization"] = vis_image

                        results.append(result)

                    batch = []
                    orig_sizes = []

            except Exception as e:
                logger.error(f"Error processing image at index {idx}: {str(e)}")
                result = {
                    "boxes": np.array([]),
                    "labels": np.array([]),
                    "scores": np.array([]),
                }
                if visualize:
                    result["visualization"] = None
                results.append(result)

        logger.success(f"Batch processing complete. Processed {len(results)} images")
        return results