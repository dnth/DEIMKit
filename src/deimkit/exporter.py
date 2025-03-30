import os
import torch
import torch.nn as nn
from loguru import logger
from typing import Any, Dict, Optional, Tuple


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
        input_shape: Optional[Tuple[int, int, int, int]] = None,
        check: bool = True,
        simplify: bool = True,
        dynamic_batch: bool = True,
        fp16: bool = False,
        opset_version: int = 20,
        device: Optional[str] = None,
    ) -> str:
        """
        Export a model to ONNX format from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file (.pth)
            output_path: Path for the ONNX model (defaults to checkpoint_path with .onnx extension)
            input_shape: Shape of the input tensor (batch_size, channels, height, width).
                         If None, will be determined from the config.
            check: Whether to validate the exported ONNX model
            simplify: Whether to simplify the exported ONNX model
            dynamic_batch: Whether to allow dynamic batch size in the exported model.
            fp16: Whether to export the model in FP16 precision (requires CUDA).
            opset_version: The ONNX opset version to use for export.
            device: The device to use for export ('cpu' or 'cuda'). Auto-selected if None.

        Returns:
            Path to the exported ONNX model
        """
        # Determine device
        _device_str = device if device else ("cuda" if fp16 else "cpu")
        if fp16 and _device_str == "cpu":
            logger.warning("FP16 export requested but device is CPU. Switching to CUDA.")
            _device_str = "cuda"
        if _device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA device requested but not available. Switching to CPU.")
            _device_str = "cpu"
            if fp16:
                logger.warning("FP16 export disabled as CUDA is not available.")
                fp16 = False

        _device = torch.device(_device_str)
        logger.info(f"Using device: {_device}")

        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=_device)

        # Extract state dict
        if "ema" in checkpoint and "module" in checkpoint["ema"]:
            logger.info("Using EMA weights for model export")
            state_dict = checkpoint["ema"]["module"]
        else:
            logger.info("EMA weights not found, using regular model weights")
            state_dict = checkpoint.get("model", checkpoint.get("state_dict"))
        if state_dict is None:
            logger.error("Could not find model state_dict in checkpoint.")
            raise KeyError("Checkpoint does not contain 'model' or 'state_dict' key.")

        # Load state dict into model
        self.config.model.load_state_dict(state_dict)

        # Convert model to deploy mode
        model = self.config.model.deploy()
        postprocessor = self.config.postprocessor.deploy()

        # Determine output path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
            suffix = ""
            if dynamic_batch:
                suffix += "_n_batch"
            if fp16:
                suffix += "_fp16"
            output_path = f"{base_name}{suffix}.onnx"
        else:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Create wrapper model and move to device
        wrapper_model = self._create_wrapper_model(model, postprocessor).to(_device)
        wrapper_model.eval()

        # Determine input shape from config if not provided
        if input_shape is None:
            logger.info("Input shape not provided, getting size from config")
            base_size = self.config.get(
                "yaml_cfg.train_dataloader.collate_fn.base_size", None
            ) or self.config.get(
                 "yaml_cfg.val_dataloader.collate_fn.base_size", None
            )

            if base_size is None:
                logger.warning(
                    "Base size not found in config (checked train/val dataloader.collate_fn.base_size)."
                    " Please specify input_shape explicitly."
                )
                raise ValueError(
                    "Could not determine input shape from config. Please provide input_shape parameter."
                )

            if isinstance(base_size, (list, tuple)) and len(base_size) == 2:
                height, width = base_size
            elif isinstance(base_size, int):
                height, width = base_size, base_size
            else:
                 logger.error(f"Unexpected base_size format in config: {base_size}")
                 raise ValueError("Invalid base_size format in config.")

            input_shape = (1, 3, height, width)
            logger.info(f"Using input shape from config: {input_shape}")

        # Create dummy inputs and move to device
        dummy_data = torch.rand(*input_shape, device=_device)
        dummy_size = torch.tensor([[input_shape[2], input_shape[3]]], device=_device)
        if not dynamic_batch and input_shape[0] > 1:
             dummy_size = dummy_size.repeat(input_shape[0], 1)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Define dynamic axes
        _dynamic_axes = None
        if dynamic_batch:
            _dynamic_axes = {
                "images": {0: "N"},
                "orig_target_sizes": {0: "N"},
                "labels": {0: "N"},
                "boxes": {0: "N"},
                "scores": {0: "N"},
            }
        logger.info(f"Using dynamic axes: {_dynamic_axes}")

        logger.info(f"Exporting model to ONNX: {output_path}")

        try:
            # Export to ONNX with FP16 context if enabled
            export_kwargs = {
                "model": wrapper_model,
                "args": (dummy_data, dummy_size),
                "f": output_path,
                "input_names": ["images", "orig_target_sizes"],
                "output_names": ["labels", "boxes", "scores"],
                "dynamic_axes": _dynamic_axes,
                "opset_version": opset_version,
                "do_constant_folding": True,
            }

            if fp16:
                with torch.autocast(device_type=_device_str, dtype=torch.float16):
                    _ = wrapper_model(dummy_data, dummy_size)
                    torch.onnx.export(**export_kwargs)
            else:
                torch.onnx.export(**export_kwargs)

            logger.success(f"ONNX export completed successfully: {output_path}")

            # Validate and simplify if requested
            final_output_path = output_path

            if simplify:
                simplified_path = self._simplify_onnx_model(
                    output_path,
                    input_shapes={
                        "images": dummy_data.shape,
                        "orig_target_sizes": dummy_size.shape,
                    },
                    target_path=output_path,
                )
                if simplified_path:
                     final_output_path = simplified_path

            if check:
                self._check_onnx_model(final_output_path)

            return final_output_path

        except Exception as e:
            logger.error(f"ONNX export failed: {str(e)}", exc_info=True)
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                    logger.info(f"Removed partially exported file: {output_path}")
                except OSError as remove_err:
                    logger.warning(f"Failed to remove partial file {output_path}: {remove_err}")
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
        if not os.path.exists(model_path):
             logger.error(f"Cannot check ONNX model: File not found at {model_path}")
             return
        try:
            import onnx

            onnx_model = onnx.load(model_path)
            onnx.checker.check_model(onnx_model)
            logger.info(f"ONNX model validation successful: {model_path}")
        except ImportError:
            logger.warning("ONNX validation skipped: 'onnx' package not installed")
        except Exception as e:
            logger.error(f"ONNX model validation failed for {model_path}: {str(e)}", exc_info=True)

    def _simplify_onnx_model(
        self,
        model_path: str,
        input_shapes: dict[str, tuple],
        target_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Simplify the exported ONNX model using onnxsim.

        Args:
            model_path: Path to the input ONNX model.
            input_shapes: Dictionary mapping input names to their shapes for simplification.
            target_path: Path to save the simplified model. If None, saves inplace (overwrites model_path).

        Returns:
            Path to the simplified model, or None if simplification failed.
        """
        if not os.path.exists(model_path):
             logger.error(f"Cannot simplify ONNX model: File not found at {model_path}")
             return None
        if target_path is None:
            target_path = model_path
        try:
            import onnx
            import onnxsim

            logger.info(f"Simplifying ONNX model: {model_path} -> {target_path}")
            onnx_model_simplify, check = onnxsim.simplify(
                model_path,
                test_input_shapes=input_shapes,
                perform_optimization=True,
                skip_fuse_bn=False,
            )

            if check:
                onnx.save(onnx_model_simplify, target_path)
                logger.success(f"ONNX model simplification successful: {target_path}")
                return target_path
            else:
                logger.error(f"ONNX model simplification check failed for: {model_path}")
                if model_path != target_path and os.path.exists(model_path):
                     import shutil
                     shutil.copyfile(model_path, target_path)
                     logger.warning(f"Saved original (unsimplified) model to {target_path} due to check failure.")
                     return target_path
                return None

        except ImportError:
            logger.warning(
                "ONNX simplification skipped: 'onnx' or 'onnxsim' package not installed"
            )
            return None
        except Exception as e:
            logger.error(f"ONNX model simplification failed for {model_path}: {str(e)}", exc_info=True)
            if model_path != target_path and os.path.exists(model_path):
                 import shutil
                 try:
                      shutil.copyfile(model_path, target_path)
                      logger.warning(f"Saved original model to {target_path} due to simplification error.")
                      return target_path
                 except Exception as copy_e:
                      logger.error(f"Failed to copy original model {model_path} to {target_path}: {copy_e}")

            return None
