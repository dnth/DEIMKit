import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
from typing import Dict, List, Union, Tuple, Optional

# COCO class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
    'toothbrush'
]

def visualize_detections(
    image: Union[str, Image.Image, np.ndarray],
    detections: Dict[str, np.ndarray],
    class_names: List[str] = COCO_CLASSES,
    score_threshold: float = 0.25,
    figsize: Tuple[int, int] = (12, 12),
    save_path: Optional[str] = None,
    show: bool = True,
    colors: Optional[Dict[int, Tuple[float, float, float]]] = None
) -> plt.Figure:
    """
    Visualize object detection results on an image.
    
    Args:
        image: PIL Image, numpy array, or path to image file
        detections: Dictionary with 'boxes', 'labels', and 'scores' keys
        class_names: List of class names corresponding to label indices
        score_threshold: Minimum confidence score to display
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the visualization
        show: Whether to display the plot
        colors: Optional dictionary mapping class indices to RGB colors (0-1 range)
        
    Returns:
        Matplotlib figure object
    """
    # Load image if it's a path
    if isinstance(image, str):
        image = Image.open(image)
    
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)
    
    # Get detection data
    boxes = detections['boxes']
    labels = detections['labels']
    scores = detections['scores']
    
    # Filter by score threshold
    if score_threshold > 0 and len(scores) > 0:
        mask = scores >= score_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
    
    # Generate colors for classes if not provided
    if colors is None:
        colors = {}
        for i in range(len(class_names)):
            # Use HSV color space with reduced value (brightness) for better contrast
            hue = i / len(class_names)
            colors[i] = plt.cm.hsv(hue, 0.8, 0.7)[:3]  # Reduced saturation and value
    
    # Draw each detection
    for box, label, score in zip(boxes, labels, scores):
        # Get coordinates - handle both [x1, y1, x2, y2] and [y1, x1, y2, x2] formats
        if len(box) == 4:
            # Check if the box format is [y1, x1, y2, x2] (COCO format)
            if 'image_size' in detections and detections.get('box_format', '') == 'yxyx':
                y1, x1, y2, x2 = box
                width = x2 - x1
                height = y2 - y1
                x1, y1 = x1, y1
            else:  # Default to [x1, y1, x2, y2] format
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
        else:
            # Handle other formats or raise error
            raise ValueError(f"Unexpected box format with {len(box)} values")
        
        # Get class name and color
        label_idx = int(label)
        class_name = class_names[label_idx] if label_idx < len(class_names) else f"Class {label_idx}"
        color = colors.get(label_idx, (1.0, 0, 0))  # Default to red if color not found
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x1, y1), width, height, 
            linewidth=2, 
            edgecolor=color, 
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label and score
        label_text = f"{class_name}: {score:.2f}"
        ax.text(
            x1, y1 - 5, 
            label_text, 
            color='white', 
            fontsize=10, 
            bbox=dict(facecolor=color, alpha=0.8, pad=2)
        )
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show or close
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def draw_on_image(
    image: Union[str, Image.Image, np.ndarray],
    detections: Dict[str, np.ndarray],
    class_names: List[str] = COCO_CLASSES,
    score_threshold: float = 0.25,
    output_path: Optional[str] = None,
    box_thickness: int = 2,
    text_size: float = 1.0,
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None
) -> Image.Image:
    """
    Draw detection results directly on the image and return the modified image.
    
    Args:
        image: PIL Image, numpy array, or path to image file
        detections: Dictionary with 'boxes', 'labels', and 'scores' keys
        class_names: List of class names corresponding to label indices
        score_threshold: Minimum confidence score to display
        output_path: Optional path to save the output image
        box_thickness: Thickness of bounding box lines
        text_size: Size of the text relative to default
        colors: Optional dictionary mapping class indices to BGR colors (0-255 range)
        
    Returns:
        PIL Image with detections drawn on it
    """
    import cv2  # Import here to avoid dependency for those who don't need this function
    
    # Load image if it's a path
    if isinstance(image, str):
        image = Image.open(image)
    
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image.copy()
    
    # Convert RGB to BGR for OpenCV
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Get detection data
    boxes = detections['boxes']
    labels = detections['labels']
    scores = detections['scores']
    
    # Filter by score threshold
    if score_threshold > 0 and len(scores) > 0:
        mask = scores >= score_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
    
    # Generate colors for classes if not provided
    if colors is None:
        colors = {}
        for i in range(len(class_names)):
            # Convert HSV to BGR for OpenCV, using darker colors
            hsv_color = np.array([[[
                i / len(class_names) * 179,  # Hue
                200,  # Reduced saturation
                180   # Reduced value/brightness
            ]]], dtype=np.uint8)
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            colors[i] = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))
    
    # Draw each detection
    for box, label, score in zip(boxes, labels, scores):
        # Get coordinates - handle both [x1, y1, x2, y2] and [y1, x1, y2, x2] formats
        if len(box) == 4:
            # Check if the box format is [y1, x1, y2, x2] (COCO format)
            if 'image_size' in detections and detections.get('box_format', '') == 'yxyx':
                y1, x1, y2, x2 = [int(coord) for coord in box]
            else:  # Default to [x1, y1, x2, y2] format
                x1, y1, x2, y2 = [int(coord) for coord in box]
        else:
            # Handle other formats or raise error
            raise ValueError(f"Unexpected box format with {len(box)} values")
        
        # Get class name and color
        label_idx = int(label)
        class_name = class_names[label_idx] if label_idx < len(class_names) else f"Class {label_idx}"
        color = colors.get(label_idx, (0, 0, 255))  # Default to red if color not found
        
        # Draw rectangle
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, box_thickness)
        
        # Add label and score
        label_text = f"{class_name}: {score:.2f}"
        text_size_value = 0.5 * text_size
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, text_size_value, 1
        )
        
        # Draw text background
        cv2.rectangle(
            image_cv, 
            (x1, y1 - text_height - 5), 
            (x1 + text_width, y1), 
            color, 
            -1
        )
        
        # Draw text
        cv2.putText(
            image_cv, 
            label_text, 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            text_size_value, 
            (255, 255, 255), 
            1, 
            cv2.LINE_AA
        )
    
    # Convert back to RGB and then to PIL Image
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    result_image = Image.fromarray(image_rgb)
    
    # Save if path provided
    if output_path:
        result_image.save(output_path)
    
    return result_image