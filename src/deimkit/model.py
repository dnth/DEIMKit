from typing import Any, Dict, Optional

from loguru import logger

from .config import Config


def configure_model(
    config: Config,
    num_queries: Optional[int] = None,
    pretrained: Optional[bool] = None,
    freeze_at: Optional[int] = None,
) -> Config:
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
    if not isinstance(config, Config):
        raise TypeError(
            f"Expected a deimkit.config.Config object, but got {type(config)}"
        )

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

    if num_queries is not None:
        if decoder_type:
            key = f"yaml_cfg.{decoder_type}.num_queries"
            updates[key] = num_queries
            logger.info(f"Setting '{key}' to: {num_queries}")

            # Sets the number of top queries to be equal to the number of queries
            key = f"yaml_cfg.PostProcessor.num_top_queries"
            updates[key] = num_queries
            logger.info(f"Setting '{key}' to: {num_queries}")
        else:
            logger.warning(f"Cannot set 'num_queries' because decoder type is unknown.")

    if pretrained is not None:
        if backbone_type:
            key = f"yaml_cfg.{backbone_type}.pretrained"
            updates[key] = pretrained
            logger.info(f"Setting '{key}' to: {pretrained}")
        else:
            logger.warning(
                f"Cannot set 'use_pretrained_backbone' because backbone type is unknown."
            )

    if freeze_at is not None:
        if backbone_type:
            key = f"yaml_cfg.{backbone_type}.freeze_at"
            updates[key] = freeze_at

            if (
                freeze_at > 0
            ):  # If freeze_at is greater than 0, then set freeze_stem_only to False. freeze_at = 0 also means freeze the stem block.
                key = f"yaml_cfg.{backbone_type}.freeze_stem_only"
                updates[key] = False
                logger.info(f"Setting '{key}' to: {False}")

            logger.info(f"Setting '{key}' to: {freeze_at}")
        else:
            logger.warning(f"Cannot set 'freeze_at' because backbone type is unknown.")

    if updates:
        config.update(**updates)  # Modifies the original config object

    return config  # Return the same object
