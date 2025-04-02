import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from typing import Any, Dict, Optional, Tuple


class Exporter:
    """
    Export a DEIM model to ONNX format.

    This class provides functionality to export trained DEIM models to ONNX format
    for deployment in production environments, optionally including preprocessing steps.
    """

    def __init__(self, config: Any):
        """
        Initialize the exporter with a model configuration.

        Args:
            config: Configuration object containing model configuration
        """
        self.config = config

    class PreprocessingModule(nn.Module):
        """Handles image preprocessing: resize, BGR->RGB, normalize."""
        def __init__(self, target_height: int, target_width: int):
            super().__init__()
            self.target_height = target_height
            self.target_width = target_width
            logger.info(
                f"Initialized PreprocessingModule with target size: "
                f"({target_height}, {target_width})"
            )

        def forward(self, input_bgr: torch.Tensor) -> torch.Tensor:
            """
            Apply preprocessing steps.

            Args:
                input_bgr: Input tensor in BGR format (N, 3, H, W).

            Returns:
                Preprocessed tensor in RGB format, normalized, and resized.
            """
            # 1. Resize
            x = F.interpolate(
                input=input_bgr,
                size=(self.target_height, self.target_width),
                mode='bilinear', # Common interpolation mode, adjust if needed
                align_corners=False # Common practice
            )
            logger.debug(f"Preprocessing: Resized shape: {x.shape}")

            # 2. BGR -> RGB
            # Ensure input has 3 channels
            if x.shape[1] != 3:
                 raise ValueError(f"Input tensor must have 3 channels (BGR), got {x.shape[1]}")
            # Swap channels: (B:0, G:1, R:2) -> (R:2, G:1, B:0)
            x = x[:, [2, 1, 0], :, :]
            logger.debug("Preprocessing: Swapped BGR to RGB")

            # 3. Normalize (0-255 -> 0-1)
            # Assuming input is uint8 [0, 255], scale to [0, 1]
            x = x * (1.0 / 255.0)
            logger.debug("Preprocessing: Normalized pixel values to [0, 1]")

            return x

    def to_onnx(
        self,
        checkpoint_path: str,
        output_path: str | None = None,
        input_shape: Optional[Tuple[int, int, int, int]] = None,
        check: bool = True,
        simplify: bool = True,
        dynamic_batch: bool = True,
        dynamic_input_size: bool = True,
        include_preprocessing: bool = True,
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
                         If `include_preprocessing` is True, this defines the *target* size
                         for the internal resize operation. Otherwise, it's the direct model input size.
                         If None, will be determined from the config.
            check: Whether to validate the exported ONNX model
            simplify: Whether to simplify the exported ONNX model
            dynamic_batch: Whether to allow dynamic batch size (N) in the exported model.
            dynamic_input_size: Whether to allow dynamic input height (H) and width (W)
                                if `include_preprocessing` is True.
            include_preprocessing: If True, include resize, BGR->RGB, and normalization
                                   steps in the exported ONNX graph. The input will
                                   expect raw BGR images.
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

        # Determine input shape from config if not provided
        target_height, target_width = None, None
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

            target_height, target_width = height, width
            input_shape = (1, 3, target_height, target_width)
            logger.info(f"Using target shape from config: {input_shape}")
        else:
             # If input_shape is provided, extract target height/width
             _, _, target_height, target_width = input_shape
             logger.info(f"Using provided target shape: {(target_height, target_width)}")

        # Create preprocessing module if requested
        preprocessing_module = None
        if include_preprocessing:
             if target_height is None or target_width is None:
                 raise ValueError("Cannot include preprocessing without a defined target height/width."
                                  "Provide input_shape or ensure base_size is in config.")
             preprocessing_module = self.PreprocessingModule(target_height, target_width)
             logger.info("Including preprocessing steps in the ONNX model.")

        # Create wrapper model and move to device
        wrapper_model = self._create_wrapper_model(
            model, postprocessor, preprocessing_module
        ).to(_device)
        wrapper_model.eval()

        # Create dummy inputs and move to device
        dummy_data_shape = input_shape
        dummy_data = torch.rand(*dummy_data_shape, device=_device)

        # The 'orig_target_sizes' input typically corresponds to the size *before* padding/resizing
        # If preprocessing is included, this might represent the original image size.
        # If not, it represents the size the postprocessor needs.
        # For export, we use the target H/W here. The actual value depends on runtime usage.
        dummy_size_h, dummy_size_w = target_height, target_width
        # If include_preprocessing and dynamic input size, the actual input H/W can vary.
        # For export dummy data, we still need concrete values.
        dummy_size = torch.tensor([[dummy_size_h, dummy_size_w]], device=_device)
        if not dynamic_batch and dummy_data_shape[0] > 1:
             dummy_size = dummy_size.repeat(dummy_data_shape[0], 1)

        # Define input/output names based on whether preprocessing is included
        input_names = ["images", "orig_target_sizes"]
        if include_preprocessing:
            input_names[0] = "input_bgr"

        output_names = ["labels", "boxes", "scores"]

        # Define dynamic axes
        _dynamic_axes = {}
        first_input_name = input_names[0]
        _dynamic_axes[first_input_name] = {}
        _dynamic_axes["orig_target_sizes"] = {}
        _dynamic_axes["labels"] = {}
        _dynamic_axes["boxes"] = {}
        _dynamic_axes["scores"] = {}

        if dynamic_batch:
            _dynamic_axes[first_input_name][0] = "N"
            _dynamic_axes["orig_target_sizes"][0] = "N"
            _dynamic_axes["labels"][0] = "N"
            _dynamic_axes["boxes"][0] = "N"
            _dynamic_axes["scores"][0] = "N"

        # Add dynamic H/W for the input if preprocessing is enabled and requested
        if include_preprocessing and dynamic_input_size:
            _dynamic_axes[first_input_name][2] = "H"
            _dynamic_axes[first_input_name][3] = "W"
            # Note: 'orig_target_sizes' might also need dynamic axes depending on usage,
            # but typically it relates to the *original* size before preprocessing.
            # Keeping it simple here unless specific needs arise.

        # Remove empty dicts if no dynamic axes are specified for a name
        _dynamic_axes = {k: v for k, v in _dynamic_axes.items() if v}
        if not _dynamic_axes:
             _dynamic_axes = None

        logger.info(f"Using input names: {input_names}")
        logger.info(f"Using output names: {output_names}")
        logger.info(f"Using dynamic axes: {_dynamic_axes}")
        logger.info(f"Exporting model to ONNX: {output_path}")

        try:
            # Export to ONNX with FP16 context if enabled
            export_kwargs = {
                "model": wrapper_model,
                "args": (dummy_data, dummy_size),
                "f": output_path,
                "input_names": input_names,
                "output_names": output_names,
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
                # Input shapes for simplification should match the dummy data used for export
                input_shapes_for_sim = {
                    input_names[0]: dummy_data.shape,
                    input_names[1]: dummy_size.shape,
                }
                logger.info(f"Simplifying ONNX model with input shapes: {input_shapes_for_sim}")
                simplified_path = self._simplify_onnx_model(
                    output_path,
                    input_shapes=input_shapes_for_sim,
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
        self,
        model: nn.Module,
        postprocessor: nn.Module,
        preprocessing: Optional[nn.Module] = None
    ) -> nn.Module:
        """
        Create a wrapper model that includes optional preprocessing, the main model,
        and the postprocessor.
        """

        class WrappedModel(nn.Module):
            def __init__(
                self,
                model: nn.Module,
                postprocessor: nn.Module,
                preprocessing: Optional[nn.Module] = None
            ):
                super().__init__()
                self.preprocessing = preprocessing
                self.model = model
                self.postprocessor = postprocessor

            def forward(self, images: torch.Tensor, orig_target_sizes: torch.Tensor):
                # Apply preprocessing if it exists
                if self.preprocessing:
                    x = self.preprocessing(images)
                else:
                    x = images

                # Pass preprocessed data to the main model
                outputs = self.model(x)

                # Pass model outputs and original sizes to postprocessor
                return self.postprocessor(outputs, orig_target_sizes)

        return WrappedModel(model, postprocessor, preprocessing)

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
            input_shapes: Dictionary mapping input names (e.g., 'input_bgr', 'orig_target_sizes')
                          to their concrete shapes for simplification.
            target_path: Path to save the simplified model. If None, saves inplace.

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
                     try:
                         shutil.copyfile(model_path, target_path)
                         logger.warning(f"Saved original (unsimplified) model to {target_path} due to check failure.")
                         return target_path
                     except Exception as copy_e:
                         logger.error(f"Failed to copy original model {model_path} to {target_path}: {copy_e}")
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
