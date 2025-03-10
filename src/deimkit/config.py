import os
import pprint

import pkg_resources

from .engine.core import BaseConfig, YAMLConfig


def list_models():
    return [
        "deim_hgnetv2_n",
        "deim_hgnetv2_s",
        "deim_hgnetv2_m",
        "deim_hgnetv2_l",
        "deim_hgnetv2_x",
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
            raise ValueError(
                f"Model {model_name} not found. Available models: {list_models()}"
            )

        # Try to find the config file in several possible locations
        possible_paths = [
            # Option 1: Look in the package's installed configs directory
            pkg_resources.resource_filename(
                "deimkit", f"configs/{model_name}_coco.yml"
            ),
            pkg_resources.resource_filename(
                "deimkit", f"configs/deim_dfine/{model_name}_coco.yml"
            ),
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

    def print(self):
        """
        Print the configuration in a readable format using pprint.
        """
        # Convert config object to dictionary
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        pprint.pprint(config_dict, indent=2, width=100)

    def update(self, **kwargs):
        """
        Update multiple configuration parameters at once.

        Args:
            **kwargs: Key-value pairs of configuration parameters to update.

        Returns:
            Config: Self for method chaining.
        """
        for key, value in kwargs.items():
            self.set(key, value)
        return self

    def get(self, key, default=None):
        """
        Get a configuration parameter value, supporting nested keys with dot notation.

        Args:
            key (str): The configuration parameter name. Use dot notation for nested keys
                       (e.g., 'yaml_cfg.DEIM.backbone').
            default: Value to return if the parameter doesn't exist.

        Returns:
            The parameter value or default if not found.
        """
        if "." not in key:
            # Simple case: top-level attribute
            return getattr(self, key, default)

        # Handle nested keys
        parts = key.split(".")
        current = getattr(self, parts[0], None)

        if current is None:
            return default

        # Navigate through the nested structure
        for part in parts[1:]:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default

        return current

    def set(self, key, value):
        """
        Set a configuration parameter, supporting nested keys with dot notation.

        Args:
            key (str): The configuration parameter name. Use dot notation for nested keys
                       (e.g., 'yaml_cfg.DEIM.backbone').
            value: The value to set.

        Returns:
            Config: Self for method chaining.
        """
        if "." not in key:
            # Simple case: top-level attribute
            setattr(self, key, value)
            return self

        # Handle nested keys
        parts = key.split(".")
        top_key = parts[0]

        # Get or create the top-level dictionary
        if not hasattr(self, top_key):
            setattr(self, top_key, {})

        current = getattr(self, top_key)

        # Ensure we're working with a dictionary
        if not isinstance(current, dict):
            raise ValueError(
                f"Cannot set nested key '{key}': '{top_key}' is not a dictionary"
            )

        # Navigate to the correct nested dictionary
        for part in parts[1:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                current[part] = {}

            current = current[part]

        # Set the value in the final dictionary
        current[parts[-1]] = value
        return self

    def save(self, filepath):
        """
        Save the current configuration to a YAML file.

        Args:
            filepath (str): Path where the configuration will be saved.

        Returns:
            Config: Self for method chaining.
        """
        # Convert config object to dictionary
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

        import yaml

        with open(filepath, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        return self

    def reset(self):
        """
        Reset configuration to default values based on the original config file.

        Returns:
            Config: A new Config instance with default values.

        Raises:
            ValueError: If the configuration wasn't loaded from a file.
        """
        if not hasattr(self, "_config_path") or self._config_path is None:
            raise ValueError(
                "Cannot reset configuration that wasn't loaded from a file"
            )

        return type(self)(self._config_path)
