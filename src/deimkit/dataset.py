from typing import List, Dict, Any, Union, Optional
from .config import Config

def configure_dataset(
    config: Config,
    image_size: List[int],
    train_ann_file: str,
    train_img_folder: str,
    val_ann_file: str,
    val_img_folder: str,
    train_batch_size: int = 16,
    val_batch_size: int = 16,
    num_classes: int = None,
    remap_mscoco: bool = False,
    output_dir: str = None
) -> Config:
    """
    Configure dataset settings in a Config object.
    
    Args:
        config: A deimkit Config object
        image_size: Image size as [height, width]
        train_ann_file: Path to training annotation file (COCO format)
        train_img_folder: Path to training images folder
        val_ann_file: Path to validation annotation file (COCO format)
        val_img_folder: Path to validation images folder
        train_batch_size: Batch size for training
        val_batch_size: Batch size for validation
        num_classes: Number of classes in the dataset
        remap_mscoco: Whether to remap MSCOCO categories
        output_dir: Output directory for saving model and logs
        
    Returns:
        Config: The updated Config object
    """
    # Configure dataset paths
    config_updates = {
        'yaml_cfg.train_dataloader.dataset.ann_file': train_ann_file,
        'yaml_cfg.train_dataloader.dataset.img_folder': train_img_folder,
        'yaml_cfg.val_dataloader.dataset.ann_file': val_ann_file,
        'yaml_cfg.val_dataloader.dataset.img_folder': val_img_folder,
        'yaml_cfg.train_dataloader.total_batch_size': train_batch_size,
        'yaml_cfg.val_dataloader.total_batch_size': val_batch_size,
    }
    
    # Add optional configurations if provided
    if num_classes is not None:
        config_updates['yaml_cfg.num_classes'] = num_classes
        config_updates['yaml_cfg.remap_mscoco_category'] = remap_mscoco
    
    if output_dir is not None:
        config_updates['output_dir'] = output_dir
    
    # Update image size in transforms
    _update_image_size(config, image_size)
    
    # Apply all other updates
    config.update(**config_updates)
    
    return config

def _update_image_size(config: Config, size: List[int]) -> None:
    """
    Update image size in all relevant transforms and config settings.
    
    Args:
        config: A deimkit Config object
        size: Image size as [height, width]
    """
    train_transforms = config.get('yaml_cfg.train_dataloader.dataset.transforms.ops')
    val_transforms = config.get('yaml_cfg.val_dataloader.dataset.transforms.ops')
    
    _update_resize_transform(train_transforms, size)
    _update_resize_transform(val_transforms, size)
    
    # Update mosaic transform if present
    _update_mosaic_transform(train_transforms, size[0])
    
    # Set the updated transforms back to the config
    config.set('yaml_cfg.train_dataloader.dataset.transforms.ops', train_transforms)
    config.set('yaml_cfg.val_dataloader.dataset.transforms.ops', val_transforms)
    config.set('yaml_cfg.eval_spatial_size', size)

def _update_resize_transform(transforms_list: List[Dict[str, Any]], 
                            new_size: List[int]) -> bool:
    """
    Update the Resize transform in a transforms list.
    
    Args:
        transforms_list: List of transforms
        new_size: New size as [height, width]
        
    Returns:
        bool: True if transform was found and updated, False otherwise
    """
    for i, transform in enumerate(transforms_list):
        if transform.get('type') == 'Resize':
            transforms_list[i]['size'] = new_size
            return True
    return False

def _update_mosaic_transform(transforms_list: List[Dict[str, Any]], 
                            output_size: int) -> bool:
    """
    Update the Mosaic transform in a transforms list.
    
    Args:
        transforms_list: List of transforms
        output_size: New output size for mosaic
        
    Returns:
        bool: True if transform was found and updated, False otherwise
    """
    for i, transform in enumerate(transforms_list):
        if transform.get('type') == 'Mosaic':
            transforms_list[i]['output_size'] = output_size
            return True
    return False