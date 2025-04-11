import os
import torch
from tqdm import tqdm
from typing import Dict, Any
from deimkit import Trainer, Config, configure_dataset
from deimkit.engine.solver.det_engine import evaluate
import os

import logging

logger = logging.getLogger("Evaluator")

class Evaluator:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.trainer = None
        self.postprocessor = None
        self.evaluator = None
        self.val_dataloader = None
        self.criterion = None
        self.model = None
        self.ema = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(self):
        """Initialize the evaluation components"""
        # Use absolute paths
        BASE_DIR = "/home/mohamed/DEIMKit"
        OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/plantdoc/deim_hgnetv2_m_30ep_416x416")
        
        model_name = "deim_hgnetv2_m"
        
        logger.info(f"Setting up evaluator for model: {model_name}")
        
        # Determine if we're using PlantDoc dataset
        is_plantdoc = "plantdoc" in self.engine_path.lower()
        
        # Create configuration
        conf = Config.from_model_name(model_name)
        
        # Configure dataset - explicitly specify dataset_type for PlantDoc
        conf = configure_dataset(
            config=conf,
            image_size=(416, 416),
            train_ann_file=os.path.join(BASE_DIR, "data/plantdoc.coco/train/_annotations.coco.json"),
            train_img_folder=os.path.join(BASE_DIR, "data/plantdoc.coco/train"),
            val_ann_file=os.path.join(BASE_DIR, "data/plantdoc.coco/test/_annotations.coco.json"),
            val_img_folder=os.path.join(BASE_DIR, "data/plantdoc.coco/test"),
            train_batch_size=1,
            val_batch_size=1,
            num_classes=31,  # PlantDoc has 31 classes
            output_dir=OUTPUT_DIR,
            dataset_type="plantdoc" if is_plantdoc else "coco"  # Explicitly set dataset type
        )
        
        # Initialize trainer for evaluation components only
        trainer = Trainer(conf)

        trainer._setup()
        
        # Store references to evaluation components
        self.trainer = trainer
        self.postprocessor = trainer.postprocessor
        self.evaluator = trainer.evaluator
        self.val_dataloader = trainer.val_dataloader
        self.criterion = trainer.criterion  # May not be needed for inference
        
        # No need for the actual model, as we're using TensorRT
        self.model = None

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on the validation dataset.

        Returns:
            Dictionary containing evaluation metrics.
        """
        logger.info("Evaluating model...")

        # Setup if not already done
        if not hasattr(self, 'trainer') or self.trainer is None:
            self.setup()

        # Check if model exists
        if self.model is None:
            logger.error("No model provided for evaluation")
            return {"error": "No model provided for evaluation"}

        # Make sure model is in eval mode
        if hasattr(self.model, 'eval'):
            self.model.eval()

        # In this case, we're using the TRT model directly
        module = self.model
        
        results = {}

        # Try to run a simplified evaluation manually
        logger.info("Attempting simplified manual evaluation...")
        
        for i, (samples, targets) in enumerate(self.val_dataloader):
            if i >= 5:  # Just try a few samples
                break
                
            try:
                samples = samples.to(self.device)
                outputs = module(samples)
                
                # TensorRT model now returns outputs directly without need for postprocessing
                # Each output in the list is already a dict with 'labels', 'boxes', 'scores'
                
                # Add to results by image ID
                for t, detection_dict in zip(targets, outputs):
                    img_id = t["image_id"].item()
                    results[img_id] = detection_dict
                    
                logger.info(f"Processed batch {i+1}/5 for simplified evaluation")
                
            except Exception as batch_e:
                logger.error(f"Error processing batch {i}: {batch_e}")
                import traceback
                logger.error(traceback.format_exc())

        eval_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
        )
        
        logger.info(f"Evaluation completed. Results: {eval_stats}")
        return eval_stats
            