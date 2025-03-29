from typing import Optional, Any, Dict

from .config import Config

from loguru import logger

def configure_model(
    config: Config,
    num_queries: Optional[int] = None,
    use_pretrained_backbone: Optional[bool] = None,
) -> Config:
    """
    Applies specific model parameter overrides to an existing Config object
    using explicit named arguments for selected parameters.

    Modifies the passed-in Config object.

    Args:
        config: The deimkit Config object to modify.
        num_queries: Number of object queries for the decoder (e.g., DFINETransformer).
        use_pretrained_backbone: Whether to load pretrained weights for the backbone.

    Returns:
        The modified Config object (the same instance passed in).

    Raises:
        ValueError: If essential parameter paths (like component types)
                    cannot be determined from the provided config object when needed.
    """
    if not isinstance(config, Config):
        raise TypeError(f"Expected a deimkit.config.Config object, but got {type(config)}")

    updates: Dict[str, Any] = {}

    # --- Determine component types from the *provided* config ---
    # These are needed to construct the correct paths for nested parameters
    model_type: Optional[str] = None
    backbone_type: Optional[str] = None
    decoder_type: Optional[str] = None
    # No need for encoder_type unless hidden_dim or similar is added back

    try:
        model_type = config.get("yaml_cfg.model")
        if model_type:
            backbone_type = config.get(f"yaml_cfg.{model_type}.backbone")
            decoder_type = config.get(f"yaml_cfg.{model_type}.decoder")
            # Check if essential types were found for the parameters we care about
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
            logger.warning("Could not determine 'yaml_cfg.model' from provided config. Nested settings might not be applied.")

    except Exception as e:
        logger.warning(
            f"Could not fully determine component types from config. "
            f"Settings might fail. Error: {e}"
        )

    # --- Apply updates based on provided arguments ---

    if num_queries is not None:
        if decoder_type:
            key = f"yaml_cfg.{decoder_type}.num_queries"
            updates[key] = num_queries
            logger.info(f"Setting '{key}' to: {num_queries}")
        else:
            logger.warning(f"Cannot set 'num_queries' because decoder type is unknown.")

    if use_pretrained_backbone is not None:
        if backbone_type:
            key = f"yaml_cfg.{backbone_type}.pretrained"
            updates[key] = use_pretrained_backbone
            logger.info(f"Setting '{key}' to: {use_pretrained_backbone}")
        else:
            logger.warning(f"Cannot set 'use_pretrained_backbone' because backbone type is unknown.")

    # Use the existing update method from your Config class
    if updates:
        config.update(**updates) # Modifies the original config object

    return config # Return the same object
