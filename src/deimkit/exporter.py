import os
import torch
import torch.nn as nn
from loguru import logger
from typing import Any, Dict


class Exporter:
    """
    Export a DEIM model to ONNX format.

    This class provides functionality to export trained DEIM models to ONNX format
    for deployment in production environments.
    """

    def __init__(self, config: Any):
        """
        Initialize the exporter with a model configuration.

        Args:
            config: Configuration object containing model configuration
        """
        self.config = config

    def to_onnx(
        self,
        checkpoint_path: str,
        output_path: str | None = None,
        input_shape: tuple[int, int, int, int] | None = None,
        check: bool = True,
        simplify: bool = True,
    ) -> None:
        """
        Export a model to ONNX format from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file (.pth)
            output_path: Path for the ONNX model (defaults to checkpoint_path with .onnx extension)
            input_shape: Shape of the input tensor (batch_size, channels, height, width).
                         If None, will be determined from the config.
            check: Whether to validate the exported ONNX model
            simplify: Whether to simplify the exported ONNX model

        Returns:
            Path to the exported ONNX model
        """
        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract state dict
        state_dict = checkpoint.get("ema", {}).get("module", checkpoint.get("model"))

        # Load state dict into model
        self.config.model.load_state_dict(state_dict)

        # Convert model to deploy mode
        model = self.config.model.deploy()
        postprocessor = self.config.postprocessor.deploy()

        # Determine output path if not provided
        if output_path is None:
            output_path = checkpoint_path.replace(".pth", ".onnx")

        # Create wrapper model
        wrapper_model = self._create_wrapper_model(model, postprocessor)

        # Determine input shape from config if not provided
        if input_shape is None:
            logger.info("Input shape not provided, getting size from config")
            # Get base size from config
            base_size = self.config.get(
                "yaml_cfg.train_dataloader.collate_fn.base_size", None
            )

            if base_size is None:
                logger.warning(
                    "Base size not found in config. Please specify input_shape explicitly."
                )
                raise ValueError(
                    "Could not determine input shape from config. Please provide input_shape parameter."
                )

            if isinstance(base_size, (list, tuple)) and len(base_size) == 2:
                height, width = base_size
            else:
                height, width = base_size, base_size

            # Default to 3 channels (RGB) and batch size of 1
            input_shape = (1, 3, height, width)
            logger.info(f"Using input shape from config: {input_shape}")

        # Create dummy inputs
        dummy_data = torch.rand(*input_shape)
        dummy_size = torch.tensor([[input_shape[2], input_shape[3]]])

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Define dynamic axes
        dynamic_axes = {"images": {0: "N"}, "orig_target_sizes": {0: "N"}}

        logger.info(f"Exporting model to ONNX: {output_path}")

        try:
            # Export to ONNX
            torch.onnx.export(
                wrapper_model,
                (dummy_data, dummy_size),
                output_path,
                input_names=["images", "orig_target_sizes"],
                output_names=["labels", "boxes", "scores"],
                dynamic_axes=dynamic_axes,
                opset_version=16,
                do_constant_folding=True,
            )

            logger.success("ONNX export completed successfully")

            # Validate and simplify if requested
            if check:
                self._check_onnx_model(output_path)

            if simplify:
                self._simplify_onnx_model(
                    output_path,
                    {"images": dummy_data.shape, "orig_target_sizes": dummy_size.shape},
                )

        except Exception as e:
            logger.error(f"ONNX export failed: {str(e)}")
            raise RuntimeError(f"Failed to export model to ONNX: {str(e)}") from e

    def _create_wrapper_model(
        self, model: nn.Module, postprocessor: nn.Module
    ) -> nn.Module:
        """Create a wrapper model that includes both model and postprocessor."""

        class WrappedModel(nn.Module):
            def __init__(self, model: nn.Module, postprocessor: nn.Module):
                super().__init__()
                self.model = model
                self.postprocessor = postprocessor

            def forward(self, images: torch.Tensor, orig_target_sizes: torch.Tensor):
                outputs = self.model(images)
                return self.postprocessor(outputs, orig_target_sizes)

        return WrappedModel(model, postprocessor)

    def _check_onnx_model(self, model_path: str) -> None:
        """Check if the exported ONNX model is valid."""
        try:
            import onnx

            onnx_model = onnx.load(model_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model validation successful")
        except ImportError:
            logger.warning("ONNX validation skipped: onnx package not installed")

    def _simplify_onnx_model(
        self, model_path: str, input_shapes: dict[str, tuple]
    ) -> None:
        """Simplify the exported ONNX model."""
        try:
            import onnx
            import onnxsim

            onnx_model_simplify, check = onnxsim.simplify(
                model_path, test_input_shapes=input_shapes
            )
            onnx.save(onnx_model_simplify, model_path)
            status = "successful" if check else "failed"
            logger.info(f"ONNX model simplification {status}")
        except ImportError:
            logger.warning(
                "ONNX simplification skipped: required packages not installed"
            )
