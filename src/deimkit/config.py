from .engine.core import YAMLConfig, BaseConfig
import os
import pkg_resources

def list_models():
    return [
        'deim_hgnetv2_n',
        'deim_hgnetv2_s',
        'deim_hgnetv2_m',
        'deim_hgnetv2_l',
        'deim_hgnetv2_x'
    ]

class Config(YAMLConfig):
    """
    Configuration class for DEIM models.
    Extends YAMLConfig to provide configuration loading and management.
    """
    def __init__(self, cfg_path=None, **kwargs):
        """
        Initialize the configuration.
        
        Args:
            cfg_path (str, optional): Path to the YAML configuration file.
            **kwargs: Additional configuration parameters to override.
        """
        if cfg_path is not None:
            super().__init__(cfg_path, **kwargs)
        else:
            # Initialize with base config if no path provided
            super(BaseConfig, self).__init__()
            
            # Apply any provided kwargs
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)
    
    @classmethod
    def from_model_name(cls, model_name, **kwargs):
        """
        Create a configuration from a predefined model name.
        
        Args:
            model_name (str): Name of the model from list_models().
            **kwargs: Additional configuration parameters to override.
            
        Returns:
            Config: Configuration instance for the specified model.
        """
        if model_name not in list_models():
            raise ValueError(f"Model {model_name} not found. Available models: {list_models()}")
    
        # Try to find the config file in several possible locations
        possible_paths = [
            # Option 1: Look in the package's installed configs directory
            pkg_resources.resource_filename('deimkit', f'configs/{model_name}_coco.yml'),
            pkg_resources.resource_filename('deimkit', f'configs/deim_dfine/{model_name}_coco.yml'),
            # Option 2: Look relative to the current working directory
            # os.path.join(os.getcwd(), f'configs/{model_name}_coco.yml'),
            # os.path.join(os.getcwd(), f'configs/deim_dfine/{model_name}_coco.yml'),
            # Option 3: Look relative to this file
            # os.path.join(os.path.dirname(os.path.abspath(__file__)), f'configs/{model_name}_coco.yml'),
            # os.path.join(os.path.dirname(os.path.abspath(__file__)), f'configs/deim_dfine/{model_name}_coco.yml'),
        ]
    
        for cfg_path in possible_paths:
            if os.path.exists(cfg_path):
                return cls(cfg_path, **kwargs)
    
        # If we get here, no config file was found
        raise FileNotFoundError(
            f"Could not find configuration file for model '{model_name}'. "
            f"Searched in: {possible_paths}"
        )
    
    def __repr__(self):
        """Return a string representation of the configuration."""
        return super().__repr__()