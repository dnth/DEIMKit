import datetime
import os
import time
from pathlib import Path
from typing import Any, Dict, Union

import torch
from loguru import logger

from .config import Config
from .engine.misc import dist_utils
from .engine.optim.lr_scheduler import FlatCosineLRScheduler
from .engine.solver import TASKS
from .engine.solver.det_engine import evaluate, train_one_epoch


class Trainer:
    """
    Trainer class for DEIM models.

    This class provides a simplified interface for training and evaluating
    DEIM models, abstracting away the complexity of the underlying implementation.
    """

    def __init__(self, config: Config):
        """
        Initialize the trainer with a configuration.

        Args:
            config: Configuration object containing model and training parameters.
        """
        self.config = config
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.postprocessor = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.evaluator = None
        self.lr_scheduler = None
        self.lr_warmup_scheduler = None
        self.ema = None
        self.scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = None
        self.last_epoch = -1

        self.distributed_initialized = False

        # Initialize process group early
        self._init_process_group()

    def _init_process_group(self) -> None:
        """Initialize the process group for distributed training."""
        if self.distributed_initialized:
            return

        # Script executed without torchrun
        if "TORCHELASTIC_RUN_ID" not in os.environ:
            logger.info("Initializing process group for single-process training")

            # Set environment variables for distributed training
            os.environ["WORLD_SIZE"] = "1"
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["MASTER_ADDR"] = (
                "127.0.0.1"  # Required for env:// initialization
            )
            os.environ["MASTER_PORT"] = "29500"  # Required for env:// initialization

            if not torch.distributed.is_initialized():
                try:
                    # Use file:// initialization which is more reliable for single-process
                    torch.distributed.init_process_group(
                        backend="gloo",
                        init_method="tcp://127.0.0.1:29500",
                        world_size=1,
                        rank=0,
                    )
                    logger.info("Process group initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize process group: {e}")

                    # Try an alternative approach using file store
                    try:
                        logger.info("Trying alternative initialization approach")
                        import tempfile

                        temp_dir = tempfile.mkdtemp()
                        file_path = os.path.join(temp_dir, "shared_file")

                        store = torch.distributed.FileStore(file_path, 1)
                        torch.distributed.init_process_group(
                            backend="gloo", store=store, rank=0, world_size=1
                        )
                        logger.info(
                            "Process group initialized successfully with FileStore"
                        )
                    except Exception as e2:
                        logger.error(f"All initialization attempts failed: {e2}")

                        # Last resort: monkey patch torch.distributed
                        logger.warning("Using monkey patching as last resort")
                        self._monkey_patch_distributed()

            self.distributed_initialized = True

        # Script executed with torchrun
        else:
            logger.info(f"Initializing process group for multi-process training")
            self.distributed_initialized = dist_utils.setup_distributed()

            rank = torch.distributed.get_rank()
            if rank != 0:
                logger.remove()

            logger.info(
                f"Distributed initialization successful: {self.distributed_initialized}"
            )

    def _monkey_patch_distributed(self):
        """Monkey patch torch.distributed functions as a last resort."""
        logger.warning("Monkey patching torch.distributed functions")

        # Save original functions
        self._original_is_initialized = torch.distributed.is_initialized
        self._original_get_rank = torch.distributed.get_rank
        self._original_get_world_size = torch.distributed.get_world_size

        # Define dummy functions
        def dummy_is_initialized():
            return True

        def dummy_get_rank():
            return 0

        def dummy_get_world_size():
            return 1

        # Patch torch.distributed functions
        torch.distributed.is_initialized = dummy_is_initialized
        torch.distributed.get_rank = dummy_get_rank
        torch.distributed.get_world_size = dummy_get_world_size

    def _setup(self) -> None:
        """Set up the training environment."""
        # Create output directory
        self.output_dir = Path(self.config.get("output_dir", "./outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Disable sync_bn and find_unused_parameters which require multi-GPU
        logger.info(
            "Disabling sync_bn and find_unused_parameters for single-process training"
        )
        self.config.sync_bn = False
        self.config.find_unused_parameters = False

        # Properly set the device in config
        self.config.device = str(
            self.device
        )  # This will set the top-level device parameter
        # write to yaml_cfg
        self.config.yaml_cfg["device"] = str(self.device)
        logger.info(f"Set device in config: {self.device}")

        # Disable multiprocessing for CPU training
        if self.device.type == "cpu":
            logger.info("Running on CPU - setting num_workers=0 for DataLoader")
            self.config.yaml_cfg["train_dataloader"]["num_workers"] = 0
            self.config.yaml_cfg["val_dataloader"]["num_workers"] = 0

        # Initialize the solver based on the task
        task = self.config.get("yaml_cfg.task", "detection")
        logger.info(f"Initializing solver for task: {task}")

        # Create the solver instance
        self.solver = TASKS[task](self.config)

        # Setup the solver for training
        try:
            self.solver.train()
        except Exception as e:
            logger.error(f"Error during solver setup: {e}")
            raise

        # Extract components from the solver
        self.model = self.solver.model
        self.criterion = self.solver.criterion
        self.postprocessor = self.solver.postprocessor
        self.optimizer = self.solver.optimizer
        self.lr_scheduler = self.solver.lr_scheduler
        self.lr_warmup_scheduler = self.solver.lr_warmup_scheduler
        self.train_dataloader = self.solver.train_dataloader
        self.val_dataloader = self.solver.val_dataloader
        self.evaluator = self.solver.evaluator
        self.ema = self.solver.ema
        self.scaler = self.solver.scaler
        self.device = self.solver.device
        self.last_epoch = self.solver.last_epoch

        logger.info(f"Training setup complete. Output directory: {self.output_dir}")

        logger.info(f"Saving config to {self.output_dir}/config.yml")
        self.config.save(f"{self.output_dir}/config.yml")

    def fit(
        self,
        epochs: int | None = None,
        flat_epoch: int | None = None,
        no_aug_epoch: int | None = None,
        warmup_iter: int | None = None,
        ema_warmups: int | None = None,
        lr: float | None = None,
        lr_gamma: float | None = None,
        weight_decay: float | None = None,
        stop_epoch: int | None = None,
        mixup_epochs: list[int] | None = None,
        save_best_only: bool = False,
        pretrained: bool = True,
    ):
        """
        Train the model according to the configuration.

        Args:
            epochs: Number of training epochs. If None, uses config value.
            flat_epoch: Number of epochs with flat learning rate. If None, uses config value.
            no_aug_epoch: Number of epochs without augmentation. If None, uses config value.
            warmup_iter: Number of warmup iterations. If None, uses config value.
            ema_warmups: Number of EMA warmup steps. If None, uses config value.
            lr: Learning rate override. If None, uses config value.
            lr_gamma: Annealing factor for the learning rate. If 0.5 (default), anneals the learning rate by 50% at the end of the training cycle.
            stop_epoch: Controls when multi-scale training should stop. A large value means continue training with multi-scale without stopping.
            mixup_epochs: List of two integers that defines the epoch range during which mixup augmentation is active.
                          If None, automatically calculated as [3%, 50%] of total epochs.
            save_best_only: If True, only save the best model checkpoint.
            pretrained: If True, use pretrained weights for the model.
        """

        logger.info("Starting training...")

        if epochs is not None:
            logger.info(f"Overriding epochs to {epochs}")
            self.config.epoches = epochs
        if flat_epoch is not None:
            logger.info(f"Overriding flat epochs to {flat_epoch}")
            self.config.flat_epoch = flat_epoch
        if warmup_iter is not None:
            logger.info(f"Overriding warmup iterations to {warmup_iter}")
            self.config.warmup_iter = warmup_iter
        if ema_warmups is not None:
            logger.info(f"Overriding EMA warmups to {ema_warmups}")
            self.config.ema_warmups = ema_warmups
        if lr is not None:
            logger.info(f"Overriding learning rate to {lr}")
            self.config.yaml_cfg["optimizer"]["lr"] = lr
        if lr_gamma is not None:
            logger.info(f"Overriding learning rate gamma to {lr_gamma}")
            self.config.yaml_cfg["lr_gamma"] = lr_gamma
            self.config.lr_gamma = lr_gamma

        if weight_decay is not None:
            logger.info(f"Overriding weight decay to {weight_decay}")
            self.config.yaml_cfg["optimizer"]["weight_decay"] = weight_decay

        # Set pretrained flag in config
        if "pretrained" in self.config.yaml_cfg.get("model", {}):
            logger.info(f"Setting pretrained flag to {pretrained}")
            self.config.yaml_cfg["model"]["pretrained"] = pretrained
        else:
            # Handle nested model configurations
            for model_key in self.config.yaml_cfg:
                if (
                    isinstance(self.config.yaml_cfg[model_key], dict)
                    and "pretrained" in self.config.yaml_cfg[model_key]
                ):
                    logger.info(
                        f"Setting pretrained flag to {pretrained} for {model_key}"
                    )
                    self.config.yaml_cfg[model_key]["pretrained"] = pretrained

        # Get training parameters
        num_epochs = self.config.get("epoches", 50)
        clip_max_norm = self.config.get("clip_max_norm", 0.1)
        print_freq = self.config.get("print_freq", 100)
        checkpoint_freq = self.config.get("checkpoint_freq", 4)

        # Calculate mixup epochs if not provided
        if mixup_epochs is None:
            # Calculate as 3% and 50% of total epochs
            start_mixup = max(1, int(num_epochs * 0.04))  # At least epoch 1
            end_mixup = int(num_epochs * 0.5)
            mixup_epochs = [start_mixup, end_mixup]
            self.config.yaml_cfg["train_dataloader"]["collate_fn"]["mixup_epochs"] = (
                mixup_epochs
            )
            logger.info(f"Automatically calculated mixup epochs: {mixup_epochs}")

            # Data Augmentation Epochs. Gradually reduce data augmentation at these epochs.
            data_aug_1 = start_mixup
            data_aug_2 = end_mixup
            data_aug_3 = int(num_epochs * 0.9)

            self.config.yaml_cfg["train_dataloader"]["dataset"]["transforms"]["policy"][
                "epoch"
            ] = [data_aug_1, data_aug_2, data_aug_3]
            logger.info(
                f"Automatically calculated data augmentation epochs: {data_aug_1}, {data_aug_2}, {data_aug_3}"
            )

        if stop_epoch is None:
            stop_epoch = int(num_epochs * 0.9)
            self.config.yaml_cfg["train_dataloader"]["collate_fn"]["stop_epoch"] = (
                stop_epoch
            )
            logger.info(f"Automatically calculated stop epoch: {stop_epoch}")

        if no_aug_epoch is None:
            no_aug_epoch = max(
                1, int(num_epochs * 0.13)
            )  # No augmentation epochs of 13% from total epochs
            self.config.no_aug_epoch = no_aug_epoch
            self.config.yaml_cfg["no_aug_epoch"] = no_aug_epoch
            logger.info(
                f"Automatically calculated no augmentation epochs: {no_aug_epoch}"
            )
        else:
            logger.info(f"Using provided no augmentation epochs: {no_aug_epoch}")
            self.config.no_aug_epoch = no_aug_epoch

        if flat_epoch is None:
            flat_epoch = max(
                1, int(num_epochs * 0.5)
            )  # Half of total epochs, at least 1
            self.config.flat_epoch = flat_epoch
            self.config.yaml_cfg["flat_epoch"] = flat_epoch
            logger.info(f"Automatically calculated flat epochs: {flat_epoch}")
        else:
            logger.info(f"Using provided flat epochs: {flat_epoch}")

        if warmup_iter is None:
            # Get number of images from train folder
            num_images = len(
                os.listdir(
                    self.config.yaml_cfg["train_dataloader"]["dataset"]["img_folder"]
                )
            )

            # Calculate num iterations per epoch - num_images / batch_size
            iter_per_epoch = (
                num_images
                / self.config.yaml_cfg["train_dataloader"]["total_batch_size"]
            )

            # Scale warmup iterations based on total epochs (approximately 5% of total iterations)
            warmup_iter = int(iter_per_epoch * num_epochs * 0.05)

            # Ensure a minimum value of 1 epoch worth of iterations
            min_warmup_iter = int(iter_per_epoch)
            warmup_iter = max(warmup_iter, min_warmup_iter)

            # Set warmup_iter to that
            self.config.warmup_iter = warmup_iter
            logger.info(
                f"Automatically calculated warmup iterations: {warmup_iter} ({warmup_iter / iter_per_epoch:.1f} epochs)"
            )

        if ema_warmups is None:
            # Scale EMA warmups based on total epochs (approximately 5% of total iterations)
            ema_warmups = int(iter_per_epoch * num_epochs * 0.05)

            # Ensure a minimum value of 1 epoch worth of iterations
            min_ema_warmups = int(iter_per_epoch)
            ema_warmups = max(ema_warmups, min_ema_warmups)

            self.config.ema_warmups = ema_warmups
            logger.info(
                f"Automatically calculated EMA warmups: {ema_warmups} ({ema_warmups / iter_per_epoch:.1f} epochs)"
            )

        self._setup()  # Sends all configs to the solver

        # Add device information to config
        self.config.yaml_cfg["device"] = str(self.device)
        logger.info(f"Using device: {self.device}")

        # Start training
        start_time = time.time()
        start_epoch = self.last_epoch + 1

        # Save initial model as best.pth
        if self.output_dir:
            self._save_checkpoint(0, {}, self.output_dir / "best.pth")
            logger.info(f"Initial model saved as best.pth")

        # Initialize tracking variables
        best_stats = {"epoch": 0}
        top1 = 0

        # Log model parameters
        n_parameters = sum(
            [p.numel() for p in self.model.parameters() if p.requires_grad]
        )
        logger.info(f"Number of trainable parameters: {n_parameters}")

        # Setup custom LR scheduler if specified
        self_lr_scheduler = False
        if self.config.get("lrsheduler") is not None:
            iter_per_epoch = len(self.train_dataloader)
            logger.info(f"Using custom scheduler: {self.config.get('lrsheduler')}")
            self.lr_scheduler = FlatCosineLRScheduler(
                self.optimizer,
                self.config.get("lr_gamma", 0.5),
                iter_per_epoch,
                total_epochs=num_epochs,
                warmup_iter=self.config.get("warmup_iter", 2000),
                flat_epochs=self.config.get("flat_epoch", 29),
                no_aug_epochs=self.config.get("no_aug_epoch", 8),
            )
            self_lr_scheduler = True

        for epoch in range(start_epoch, num_epochs):
            logger.info(f"Epoch {epoch}/{num_epochs}")
            # Set epoch for data loader
            if hasattr(self.train_dataloader, "set_epoch"):
                self.train_dataloader.set_epoch(epoch)

            # Train for one epoch
            train_stats = train_one_epoch(
                self_lr_scheduler,
                self.lr_scheduler,
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=clip_max_norm,
                print_freq=print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.solver.writer if hasattr(self.solver, "writer") else None,
            )

            # Update learning rate scheduler
            if not self_lr_scheduler:
                if (
                    self.lr_warmup_scheduler is None
                    or self.lr_warmup_scheduler.finished()
                ):
                    self.lr_scheduler.step()

            self.last_epoch += 1

            # Save checkpoint
            if (
                self.output_dir
                and (epoch + 1) % checkpoint_freq == 0
                and not save_best_only
            ):
                checkpoint_path = self.output_dir / f"checkpoint{epoch:04}.pth"
                self._save_checkpoint(epoch, train_stats, checkpoint_path)

            # Evaluate
            # Calculate global step for tensorboard logging
            global_step = (epoch + 1) * len(self.train_dataloader)
            writer = self.solver.writer if hasattr(self.solver, "writer") else None

            # Pass writer and global_step to evaluate
            eval_stats, _ = evaluate(
                self.ema.module if self.ema else self.model,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                writer=writer,
                global_step=global_step,
            )

            # Update best stats
            for k in eval_stats:
                if (
                    k == "coco_eval_bbox"
                    and isinstance(eval_stats[k], list)
                    and len(eval_stats[k]) > 0
                ):
                    # Handle coco_eval_bbox specially
                    map_value = eval_stats[k][0]  # Take the first value as mAP
                    if k in best_stats:
                        if map_value > best_stats[k]:
                            best_stats["epoch"] = epoch
                            best_stats[k] = map_value
                    else:
                        best_stats["epoch"] = epoch
                        best_stats[k] = map_value

                    if best_stats[k] > top1:
                        top1 = best_stats[k]
                        if self.output_dir:
                            self._save_checkpoint(
                                epoch, eval_stats, self.output_dir / "best.pth"
                            )
                            logger.info(
                                f"🏆 NEW BEST MODEL! Epoch {epoch} / mAP: {best_stats[k]}"
                            )
                elif k != "coco_eval_bbox":
                    # Handle other metrics
                    if k in best_stats:
                        if eval_stats[k] > best_stats[k]:
                            best_stats["epoch"] = epoch
                            best_stats[k] = eval_stats[k]
                    else:
                        best_stats["epoch"] = epoch
                        best_stats[k] = eval_stats[k]

                    if k != "epoch" and best_stats[k] > top1:
                        top1 = best_stats[k]
                        if self.output_dir:
                            self._save_checkpoint(
                                epoch, eval_stats, self.output_dir / "best.pth"
                            )
                            logger.info(
                                f"🏆 NEW BEST MODEL! Epoch {epoch} / mAP: {best_stats[k]}"
                            )

            logger.info(f"✅ Current best stats: {best_stats}")

        # Save final checkpoint if not save_best_only
        if self.output_dir and not save_best_only:
            final_checkpoint_path = self.output_dir / f"checkpoint_final.pth"
            self._save_checkpoint(num_epochs - 1, eval_stats, final_checkpoint_path)
            logger.info(f"Final checkpoint saved to {final_checkpoint_path}")

        # Log training time
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"Training completed in {total_time_str}")
        logger.info(f"Best stats: {best_stats}")

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on the validation dataset.

        Returns:
            Dictionary containing evaluation metrics.
        """
        logger.info("Evaluating model...")

        # Setup if not already done
        if self.model is None:
            self._setup()

        # Use the EMA model if available, otherwise use the regular model
        module = self.ema.module if self.ema else self.model

        # Run evaluation
        eval_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
        )

        return eval_stats

    def _save_checkpoint(
        self, epoch: int, metrics: Dict[str, Any], checkpoint_path: Path
    ) -> None:
        """
        Save a checkpoint of the model.

        Args:
            epoch: Current epoch number.
            metrics: Evaluation metrics.
            checkpoint_path: Path to save the checkpoint.
        """

        # Create state dictionary
        state = {
            "date": datetime.datetime.now().isoformat(),
            "last_epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metrics": metrics,
        }

        # Add EMA state if available
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()

        # Save checkpoint
        torch.save(state, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(
        self, checkpoint_path: Union[str, Path], strict: bool = False
    ) -> None:
        """
        Load a model checkpoint, handling potential image size differences.

        Args:
            checkpoint_path: Path to the checkpoint file.
            strict: Whether to strictly enforce that the keys in state_dict match.
        """
        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Load checkpoint
        state = (
            torch.hub.load_state_dict_from_url(str(checkpoint_path), map_location="cpu")
            if str(checkpoint_path).startswith("http")
            else torch.load(checkpoint_path, map_location="cpu")
        )

        # Setup if not already done
        if self.model is None:
            self._setup()

        def load_state_dict_with_mismatch(model, state_dict):
            """Helper function to load state dict handling shape mismatches"""
            try:
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    logger.warning(
                        f"Missing keys: {missing}\nUnexpected keys: {unexpected}"
                    )
            except RuntimeError as e:
                logger.warning(
                    f"Shape mismatch, loading compatible parameters only: {e}"
                )
                current_state = model.state_dict()
                matched_state = {
                    k: v
                    for k, v in state_dict.items()
                    if k in current_state and current_state[k].shape == v.shape
                }
                model.load_state_dict(matched_state, strict=False)
                logger.info("Loaded parameters with matching shapes")

        # Load model state
        load_state_dict_with_mismatch(self.model, state["model"])

        # Load EMA state if available
        if "ema" in state and self.ema is not None:
            try:
                self.ema.load_state_dict(state["ema"])
            except RuntimeError:
                logger.info("Attempting to load EMA parameters with matching shapes...")
                load_state_dict_with_mismatch(self.ema.module, state["ema"]["module"])

        # Load optimizer state if available
        if "optimizer" in state and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except ValueError as e:
                logger.warning(f"Could not load optimizer state: {e}")

        # Update last epoch
        self.last_epoch = state.get("last_epoch", -1)
        logger.info(f"Loaded checkpoint from epoch {self.last_epoch}")

    def test(self) -> Dict[str, Any]:
        """
        Test the model on the validation dataset.

        Returns:
            Dictionary containing evaluation metrics.
        """
        logger.info("Testing model...")

        # Setup for evaluation if not already done
        if self.model is None:
            # Initialize the solver based on the task
            task = self.config.get("yaml_cfg.task", "detection")
            logger.info(f"Initializing solver for task: {task}")

            # Create the solver instance
            self.solver = TASKS[task](self.config)

            # Setup the solver for evaluation
            self.solver.eval()

            # Extract components from the solver
            self.model = self.solver.model
            self.criterion = self.solver.criterion
            self.postprocessor = self.solver.postprocessor
            self.val_dataloader = self.solver.val_dataloader
            self.evaluator = self.solver.evaluator
            self.ema = self.solver.ema
            self.device = self.solver.device

        # Run evaluation
        return self.evaluate()
