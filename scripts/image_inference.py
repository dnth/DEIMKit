import numpy as np
import onnxruntime as ort
import colorsys
import cv2

def generate_colors(num_classes):
    """Generate distinct colors for visualization."""
    hsv_tuples = [(x / num_classes, 0.8, 0.9) for x in range(num_classes)]
    colors = []
    for hsv in hsv_tuples:
        rgb = colorsys.hsv_to_rgb(*hsv)
        colors.append(tuple(int(255 * x) for x in rgb))
    return colors

def draw_boxes(image, labels, boxes, scores, ratio, padding, threshold=0.3, class_names=None):
    """Draw bounding boxes on the image."""
    # Generate colors for classes
    num_classes = len(class_names) if class_names else 91
    colors = generate_colors(num_classes)
    
    # Filter detections by threshold
    valid_indices = scores > threshold
    labels = labels[valid_indices]
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    
    pad_w, pad_h = padding
    
    for j, (lbl, box, score) in enumerate(zip(labels, boxes, scores)):
        # Get color for this class
        class_idx = int(lbl)
        color = colors[class_idx % len(colors)]
        
        # Adjust bounding box coordinates
        box_coords = [
            int((box[0] - pad_w) / ratio),
            int((box[1] - pad_h) / ratio),
            int((box[2] - pad_w) / ratio),
            int((box[3] - pad_h) / ratio),
        ]
        
        # Draw rectangle
        cv2.rectangle(image, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), color, 2)
        
        # Prepare label text
        if class_names and class_idx < len(class_names):
            label_text = f"{class_names[class_idx]} {score:.2f}"
        else:
            label_text = f"Class {class_idx} {score:.2f}"
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Draw text background
        cv2.rectangle(
            image,
            (box_coords[0], box_coords[1] - text_height - 4),
            (box_coords[0] + text_width + 4, box_coords[1]),
            color,
            -1  # Filled rectangle
        )
        
        # Calculate text color based on background brightness
        brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        
        # Draw text
        cv2.putText(
            image,
            label_text,
            (box_coords[0] + 2, box_coords[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2
        )
    
    return image

def run_inference(model_path, image_path, class_names_path=None, input_size=640):
    """Run object detection inference on an image using ONNX model."""
    # Load ONNX model
    print(f"Loading ONNX model from {model_path}...")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print(f"Using provider: {session.get_providers()[0]}")
    
    # Load class names if provided
    class_names = None
    if class_names_path:
        try:
            with open(class_names_path, "r") as f:
                class_names = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(class_names)} class names")
        except Exception as e:
            print(f"Error loading class names: {e}")
    
    # Load image
    image = cv2.imread(image_path)  # Load as BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    original_image = image.copy()
    
    # Prepare input tensor - just transpose, no normalization needed
    im_data = np.ascontiguousarray(
        image.transpose(2, 0, 1),  # HWC to CHW format
        dtype=np.float32,
    )
    im_data = np.expand_dims(im_data, axis=0)  # Add batch dimension
    orig_size = np.array([[image.shape[0], image.shape[1]]], dtype=np.int64)
    
    # Get input name from model metadata
    input_name = session.get_inputs()[0].name
    
    # Run inference
    outputs = session.run(
        output_names=None,
        input_feed={input_name: im_data, "orig_target_sizes": orig_size},
    )
    
    # Process outputs
    labels, boxes, scores = outputs

    # print(outputs)
    
    # Draw bounding boxes on the image
    result_image = draw_boxes(
        original_image, 
        labels[0], 
        boxes[0], 
        scores[0],
        1.0,  # No ratio needed since we're not resizing
        (0, 0),  # No padding needed
        class_names=class_names
    )
    
    # Save and show result
    output_path = "detection_result.jpg"
    result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
    cv2.imwrite(output_path, result_bgr)
    print(f"Detection complete. Result saved to {output_path}")
    
    # Display the result
    cv2.imshow("Detection Result", result_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return result_image

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple ONNX object detection")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model file")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--classes", type=str, help="Path to class names file (optional)")
    parser.add_argument("--size", type=int, default=640, help="Input size for model (default: 640)")
    
    args = parser.parse_args()
    
    run_inference(args.model, args.image, args.classes, args.size)