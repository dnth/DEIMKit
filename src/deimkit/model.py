from typing import TYPE_CHECKING, Any, Dict, Optional
import os
from pathlib import Path

from loguru import logger

if TYPE_CHECKING:
    from .config import Config


def list_models():
    return [
        "deim_hgnetv2_n",
        "deim_hgnetv2_s",
        "deim_hgnetv2_m",
        "deim_hgnetv2_l",
        "deim_hgnetv2_x",
    ]


def configure_model(
    config: "Config",
    num_queries: Optional[int] = None,
    pretrained: Optional[bool] = None,
    freeze_at: Optional[int] = None,
) -> "Config":
    """
    Applies specific model parameter overrides to an existing Config object
    using explicit named arguments for selected parameters.

    Modifies the passed-in Config object.

    Args:
        config: The deimkit Config object to modify.
        num_queries: Number of object queries for the decoder (e.g., DFINETransformer).
        pretrained: Whether to load pretrained weights for the backbone.
        freeze_at: Which part to freeze in the model. Default is -1 (no freezing). If 0 freeze the stem block in the backbone.
    Returns:
        The modified Config object (the same instance passed in).

    Raises:
        ValueError: If essential parameter paths (like component types)
                    cannot be determined from the provided config object when needed.
    """

    updates: Dict[str, Any] = {}

    model_type: Optional[str] = None
    backbone_type: Optional[str] = None
    decoder_type: Optional[str] = None

    try:
        model_type = config.get("yaml_cfg.model")
        if model_type:
            backbone_type = config.get(f"yaml_cfg.{model_type}.backbone")
            decoder_type = config.get(f"yaml_cfg.{model_type}.decoder")
            if not backbone_type:
                logger.warning(
                    f"Could not determine backbone type for model '{model_type}' "
                    f"from config. 'use_pretrained_backbone' setting might not be applied."
                )
            if not decoder_type:
                logger.warning(
                    f"Could not determine decoder type for model '{model_type}' "
                    f"from config. 'num_queries' setting might not be applied."
                )
        else:
            logger.warning(
                "Could not determine 'yaml_cfg.model' from provided config. Nested settings might not be applied."
            )

    except Exception as e:
        logger.warning(
            f"Could not fully determine component types from config. "
            f"Settings might fail. Error: {e}"
        )

    def update_setting(
        setting_type: str, value: Any, type_name: Optional[str], setting_path: str
    ) -> None:
        """Helper function to update settings with consistent logging"""
        if type_name:
            key = f"yaml_cfg.{type_name}.{setting_path}"
            updates[key] = value
            logger.info(f"Setting '{key}' to: {value}")
        else:
            logger.warning(
                f"Cannot set '{setting_path}' because {setting_type} type is unknown."
            )

    if num_queries is not None:
        if decoder_type:
            update_setting("decoder", num_queries, decoder_type, "num_queries")
            # Special case for PostProcessor
            update_setting(
                "post_processor", num_queries, "PostProcessor", "num_top_queries"
            )
        else:
            logger.warning("Cannot set 'num_queries' because decoder type is unknown.")

    if pretrained is not None:
        update_setting("backbone", pretrained, backbone_type, "pretrained")

    if freeze_at is not None:
        if backbone_type:
            update_setting("backbone", freeze_at, backbone_type, "freeze_at")
            if freeze_at > 0:
                update_setting("backbone", False, backbone_type, "freeze_stem_only")
        else:
            logger.warning("Cannot set 'freeze_at' because backbone type is unknown.")

    if updates:
        config.update(**updates)  

    return config  

MODEL_CHECKPOINT_URLS = {
    "deim_hgnetv2_n": "https://github.com/dnth/DEIMKit/releases/download/v0.1.0/deim_dfine_hgnetv2_n_coco_160e.pth",  # Nano model
    "deim_hgnetv2_s": "https://github.com/dnth/DEIMKit/releases/download/v0.1.0/deim_dfine_hgnetv2_s_coco_120e.pth",  # Small model
    "deim_hgnetv2_m": "https://github.com/dnth/DEIMKit/releases/download/v0.1.0/deim_dfine_hgnetv2_m_coco_90e.pth",  # Medium model
    "deim_hgnetv2_l": "https://github.com/dnth/DEIMKit/releases/download/v0.1.0/deim_dfine_hgnetv2_l_coco_50e.pth",  # Large model
    "deim_hgnetv2_x": "https://github.com/dnth/DEIMKit/releases/download/v0.1.0/deim_dfine_hgnetv2_x_coco_50e.pth",  # XLarge model
}

DEFAULT_CACHE_DIR = os.path.expanduser("./checkpoints")

def get_model_checkpoint(model_name: str, custom_checkpoint: Optional[str] = None) -> str:
    """Get the path to a model checkpoint, downloading it if necessary.
    
    Args:
        model_name: Name of the model to get checkpoint for
        custom_checkpoint: Optional path to a custom checkpoint file
        
    Returns:
        Path to the checkpoint file
    
    Raises:
        ValueError: If model_name is invalid or checkpoint cannot be found/downloaded
    """
    if model_name not in MODEL_CHECKPOINT_URLS:
        raise ValueError(f"Invalid model_name: {model_name}. Must be one of {list(MODEL_CHECKPOINT_URLS.keys())}")
        
    if custom_checkpoint is not None:
        if not os.path.exists(custom_checkpoint):
            raise ValueError(f"Custom checkpoint not found: {custom_checkpoint}")
        logger.info(f"Using custom checkpoint: {custom_checkpoint}")
        return custom_checkpoint

    return _download_checkpoint(model_name, MODEL_CHECKPOINT_URLS[model_name])

def _download_checkpoint(model_name: str, url: str) -> str:
    """Download checkpoint from URL if not already present"""
    import requests
    from tqdm import tqdm

    # Create cache directory if it doesn't exist
    cache_dir = Path(DEFAULT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Construct local path
    local_path = cache_dir / f"{model_name}.pth"

    # Download if not already present
    if not local_path.exists():
        logger.info(f"Downloading checkpoint for model {model_name}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(local_path, 'wb') as f, tqdm(
            desc=f"{model_name}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
            
        logger.success(f"Downloaded checkpoint to {local_path}")
    else:
        logger.info(f"Using cached checkpoint from {local_path}")

    return str(local_path)  
