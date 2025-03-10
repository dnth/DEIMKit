from .config import Config
import os
import torch
import torch.nn as nn
from pathlib import Path
import logging
import time
import datetime
from typing import Dict, Optional, Union, Any, List, Tuple
import functools

from .engine.misc import dist_utils
from .engine.solver import TASKS
from .engine.solver.det_engine import train_one_epoch, evaluate
from .engine.optim.lr_scheduler import FlatCosineLRScheduler
from .engine.data import CocoEvaluator


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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = None
        self.last_epoch = -1
        self.logger = self._setup_logger()
        self.distributed_initialized = False
        
        # Initialize process group early
        self._init_process_group()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for the trainer."""
        logger = logging.getLogger("DEIMTrainer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _init_process_group(self) -> None:
        """Initialize the process group for distributed training."""
        if self.distributed_initialized:
            return
            
        self.logger.info("Initializing process group for single-process training")
        
        # Set environment variables for distributed training
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "127.0.0.1"  # Required for env:// initialization
        os.environ["MASTER_PORT"] = "29500"      # Required for env:// initialization
        
        # Initialize process group
        if not torch.distributed.is_initialized():
            try:
                # Use file:// initialization which is more reliable for single-process
                torch.distributed.init_process_group(
                    backend="gloo",
                    init_method="tcp://127.0.0.1:29500",
                    world_size=1,
                    rank=0
                )
                self.logger.info("Process group initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize process group: {e}")
                
                # Try an alternative approach using file store
                try:
                    self.logger.info("Trying alternative initialization approach")
                    import tempfile
                    temp_dir = tempfile.mkdtemp()
                    file_path = os.path.join(temp_dir, "shared_file")
                    
                    store = torch.distributed.FileStore(file_path, 1)
                    torch.distributed.init_process_group(
                        backend="gloo",
                        store=store,
                        rank=0,
                        world_size=1
                    )
                    self.logger.info("Process group initialized successfully with FileStore")
                except Exception as e2:
                    self.logger.error(f"All initialization attempts failed: {e2}")
                    
                    # Last resort: monkey patch torch.distributed
                    self.logger.warning("Using monkey patching as last resort")
                    self._monkey_patch_distributed()
        
        self.distributed_initialized = True
    
    def _monkey_patch_distributed(self):
        """Monkey patch torch.distributed functions as a last resort."""
        self.logger.warning("Monkey patching torch.distributed functions")
        
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
        self.output_dir = Path(self.config.get('output_dir', './outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Modify config to avoid distributed issues
        if 'HGNetv2' in self.config.yaml_cfg:
            self.logger.info("Setting HGNetv2 pretrained to False to avoid distributed issues")
            self.config.yaml_cfg['HGNetv2']['pretrained'] = False
        
        # Disable sync_bn and find_unused_parameters which require multi-GPU
        self.logger.info("Disabling sync_bn and find_unused_parameters for single-process training")
        self.config.sync_bn = False
        self.config.find_unused_parameters = False
        
        # Initialize the solver based on the task
        task = self.config.get('yaml_cfg.task', 'detection')
        self.logger.info(f"Initializing solver for task: {task}")
        
        # Create the solver instance
        self.solver = TASKS[task](self.config)
        
        # Setup the solver for training
        try:
            self.solver.train()
        except Exception as e:
            self.logger.error(f"Error during solver setup: {e}")
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
        
        self.logger.info(f"Training setup complete. Output directory: {self.output_dir}")
        
    def train(self) -> Dict[str, Any]:
        """
        Train the model according to the configuration.
        
        Returns:
            Dictionary containing training statistics.
        """
        self.logger.info("Starting training...")
        self._setup()
        
        # Get training parameters
        num_epochs = self.config.get('epoches', 50)
        clip_max_norm = self.config.get('clip_max_norm', 0.1)
        print_freq = self.config.get('print_freq', 100)
        checkpoint_freq = self.config.get('checkpoint_freq', 4)
        
        # Training statistics
        best_stats = {'epoch': -1}
        top1 = 0
        
        # Log model parameters
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        self.logger.info(f"Number of trainable parameters: {n_parameters}")
        
        # Setup custom LR scheduler if specified
        self_lr_scheduler = False
        if self.config.get('lrsheduler') is not None:
            iter_per_epoch = len(self.train_dataloader)
            self.logger.info(f"Using custom scheduler: {self.config.get('lrsheduler')}")
            self.lr_scheduler = FlatCosineLRScheduler(
                self.optimizer, 
                self.config.get('lr_gamma', 0.5), 
                iter_per_epoch, 
                total_epochs=num_epochs,
                warmup_iter=self.config.get('warmup_iter', 2000),
                flat_epochs=self.config.get('flat_epoch', 29),
                no_aug_epochs=self.config.get('no_aug_epoch', 8)
            )
            self_lr_scheduler = True
        
        # Start training
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        
        for epoch in range(start_epoch, num_epochs):
            # Set epoch for data loader
            if hasattr(self.train_dataloader, 'set_epoch'):
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
                writer=self.solver.writer if hasattr(self.solver, 'writer') else None
            )
            
            # Update learning rate scheduler
            if not self_lr_scheduler:
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()
            
            self.last_epoch += 1
            
            # Save checkpoint
            if self.output_dir and (epoch + 1) % checkpoint_freq == 0:
                checkpoint_path = self.output_dir / f'checkpoint{epoch:04}.pth'
                self._save_checkpoint(epoch, train_stats, checkpoint_path)
            
            # Evaluate
            eval_stats = self.evaluate()
            print("================================================")
            print(eval_stats)
            print("================================================")
            
            # Update best stats
            for k in eval_stats:
                if k == 'coco_eval_bbox' and isinstance(eval_stats[k], list) and len(eval_stats[k]) > 0:
                    # Handle coco_eval_bbox specially
                    map_value = eval_stats[k][0]  # Take the first value as mAP
                    if k in best_stats:
                        if map_value > best_stats[k]:
                            best_stats['epoch'] = epoch
                            best_stats[k] = map_value
                    else:
                        best_stats['epoch'] = epoch
                        best_stats[k] = map_value
                    
                    if best_stats[k] > top1:
                        top1 = best_stats[k]
                        if self.output_dir:
                            self._save_checkpoint(epoch, eval_stats, self.output_dir / 'best.pth')
                elif k != 'coco_eval_bbox':
                    # Handle other metrics
                    if k in best_stats:
                        if eval_stats[k] > best_stats[k]:
                            best_stats['epoch'] = epoch
                            best_stats[k] = eval_stats[k]
                    else:
                        best_stats['epoch'] = epoch
                        best_stats[k] = eval_stats[k]
                    
                    if k != 'epoch' and best_stats[k] > top1:
                        top1 = best_stats[k]
                        if self.output_dir:
                            self._save_checkpoint(epoch, eval_stats, self.output_dir / 'best.pth')
            
            # Get mAP value safely from eval_stats
            # The first value in coco_eval_bbox is the AP@IoU=0.5:0.95 (primary metric)
            coco_map = eval_stats.get('coco_eval_bbox', [0.0])[0] if isinstance(eval_stats.get('coco_eval_bbox', [0.0]), list) else 0.0
            
            self.logger.info(f"Epoch {epoch} - Train loss: {train_stats['loss']:.4f}, Eval mAP: {coco_map:.4f}")
            self.logger.info(f"Best stats: {best_stats}")
        
        # Log training time
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info(f'Training completed in {total_time_str}')
        
        return best_stats
        
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on the validation dataset.
        
        Returns:
            Dictionary containing evaluation metrics.
        """
        self.logger.info("Evaluating model...")
        
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
            self.device
        )
        
        return eval_stats
        
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, Any], checkpoint_path: Path) -> None:
        """
        Save a checkpoint of the model.
        
        Args:
            epoch: Current epoch number.
            metrics: Evaluation metrics.
            checkpoint_path: Path to save the checkpoint.
        """
        self.logger.info(f"Saving checkpoint to {checkpoint_path}")
        
        # Create state dictionary
        state = {
            'date': datetime.datetime.now().isoformat(),
            'last_epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': metrics,
        }
        
        # Add EMA state if available
        if self.ema is not None:
            state['ema'] = self.ema.state_dict()
        
        # Save checkpoint
        torch.save(state, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        checkpoint_path = Path(checkpoint_path)
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load checkpoint
        if str(checkpoint_path).startswith('http'):
            state = torch.hub.load_state_dict_from_url(str(checkpoint_path), map_location='cpu')
        else:
            state = torch.load(checkpoint_path, map_location='cpu')
        
        # Setup if not already done
        if self.model is None:
            self._setup()
        
        # Load model state
        self.model.load_state_dict(state['model'])
        
        # Load optimizer state if available
        if 'optimizer' in state and self.optimizer is not None:
            self.optimizer.load_state_dict(state['optimizer'])
        
        # Load EMA state if available
        if 'ema' in state and self.ema is not None:
            self.ema.load_state_dict(state['ema'])
        
        # Update last epoch
        if 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
        
        self.logger.info(f"Loaded checkpoint from epoch {self.last_epoch}")
        
    def test(self) -> Dict[str, Any]:
        """
        Test the model on the validation dataset.
        
        Returns:
            Dictionary containing evaluation metrics.
        """
        self.logger.info("Testing model...")
        
        # Setup for evaluation if not already done
        if self.model is None:
            # Initialize the solver based on the task
            task = self.config.get('yaml_cfg.task', 'detection')
            self.logger.info(f"Initializing solver for task: {task}")
            
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
