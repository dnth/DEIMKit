import torch
import torch.nn as nn
from PIL import Image
import logging

class CustomPruner:
    def __init__(self, model, predictor):
        self.model = model
        self.predictor = predictor
        self.device = next(model.parameters()).device
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def analyze_model_structure(self):
        """Basic model structure analysis"""
        prunable_layers = []
        
        # Find all convolutional layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                prunable_layers.append((name, module))
                self.logger.info(f"Found prunable layer: {name} with shape {module.weight.shape}")
        
        return prunable_layers

    def random_prune_layer(self, layer, sparsity):
        """Randomly prune a layer's channels"""
        with torch.no_grad():
            # Get weight tensor
            weight = layer.weight.data
            
            # Calculate number of channels to prune
            num_channels = weight.shape[0]
            num_to_prune = int(num_channels * sparsity)
            
            # Randomly select channels to prune
            channels_to_prune = torch.randperm(num_channels)[:num_to_prune]
            
            # Zero out selected channels
            weight[channels_to_prune] = 0
            
            self.logger.info(f"Pruned {num_to_prune} channels from layer with shape {weight.shape}")

    def validate_pruning(self):
        """Basic validation with a test image"""
        try:
            # Load test image
            image_path = "figures/teaser_a.png"
            image = Image.open(image_path).convert("RGB")
            w, h = image.size

            # Preprocess
            im_data = self.predictor.transforms(image)
            im_data = im_data.unsqueeze(0).to(self.device)
            orig_target_sizes = torch.tensor([[w, h]])

            # Run inference
            with torch.no_grad():
                outputs = self.model(im_data, orig_target_sizes)
                self.logger.info("Validation successful - model can run inference")
                return True
                
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            return False

    def prune_model(self, sparsity=0.3):
        """Main pruning method with random pruning"""
        self.logger.info("Starting model pruning...")
        
        # 1. Find prunable layers
        prunable_layers = self.analyze_model_structure()
        
        # 2. Randomly prune each layer
        for name, layer in prunable_layers:
            self.random_prune_layer(layer, sparsity)
        
        # 3. Validate
        if self.validate_pruning():
            self.logger.info("Pruning completed successfully")
        else:
            self.logger.warning("Pruning completed but validation failed")
        
        return self.model