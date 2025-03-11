import os

import torch
from loguru import logger

def save_only_ema_weights(checkpoint_file: str) -> None:
    """Extract and save only the EMA weights."""
    logger.info(f"Loading checkpoint from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location="cpu")

    weights = {}
    if "ema" in checkpoint:
        weights["model"] = checkpoint["ema"]["module"]
    else:
        logger.error("The checkpoint does not contain 'ema'.")
        raise ValueError("The checkpoint does not contain 'ema'.")

    dir_name, base_name = os.path.split(checkpoint_file)
    name, ext = os.path.splitext(base_name)
    output_file = os.path.join(dir_name, f"{name}_ema_weights{ext}")

    torch.save(weights, output_file)
    logger.info(f"EMA weights saved to {output_file}")
