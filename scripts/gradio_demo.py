import colorsys
import os
import argparse
import time

import gradio as gr
import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image, ImageDraw
import cv2


# Use absolute paths instead of relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models/deim-blood-cell-detection_nano.onnx")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "models/classes.txt")

def generate_colors(num_classes):
    """Generate a list of distinct colors for different classes."""
    # Generate evenly spaced hues
    hsv_tuples = [(x / num_classes, 0.8, 0.9) for x in range(num_classes)]

    # Convert to RGB
    colors = []
    for hsv in hsv_tuples:
        rgb = colorsys.hsv_to_rgb(*hsv)
        # Convert to 0-255 range and to tuple
        colors.append(tuple(int(255 * x) for x in rgb))

    return colors


def draw(images, labels, boxes, scores, scales, paddings, thrh=0.4, class_names=None):
    """Draw detection boxes on images."""
    result_images = []
    colors = generate_colors(len(class_names) if class_names else 91)

    for i, im in enumerate(images):
        # Convert PIL to numpy if needed
        if isinstance(im, Image.Image):
            im = np.array(im)

        # Filter detections by threshold
        valid_indices = scores[i] > thrh
        valid_labels = labels[i][valid_indices]
        valid_boxes = boxes[i][valid_indices]
        valid_scores = scores[i][valid_indices]

        # Scale boxes from padded size to original image size
        scale = scales[i]
        x_offset, y_offset = paddings[i]
        
        valid_boxes[:, [0, 2]] = (valid_boxes[:, [0, 2]] - x_offset) / scale  # x coordinates
        valid_boxes[:, [1, 3]] = (valid_boxes[:, [1, 3]] - y_offset) / scale  # y coordinates

        # Draw boxes
        for label, box, score in zip(valid_labels, valid_boxes, valid_scores):
            class_idx = int(label)
            color = colors[class_idx % len(colors)]
            
            # Convert coordinates to integers
            box = [int(coord) for coord in box]
            
            # Draw rectangle
            cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), color, 2)

            # Prepare label text
            if class_names and class_idx < len(class_names):
                label_text = f"{class_names[class_idx]} {score:.2f}"
            else:
                label_text = f"Class {class_idx} {score:.2f}"

            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )

            # Draw text background
            cv2.rectangle(
                im,
                (box[0], box[1] - text_height - 4),
                (box[0] + text_width + 4, box[1]),
                color,
                -1,
            )

            # Calculate text color based on background brightness
            brightness = sum(color) / 3
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

            # Draw text
            cv2.putText(
                im,
                label_text,
                (box[0] + 2, box[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                text_color,
                2,
            )

        # Convert back to PIL Image
        result_images.append(Image.fromarray(im))

    return result_images


def load_model(model_path):
    """
    Load an ONNX model for inference.

    Args:
        model_path: Path to the ONNX model file

    Returns:
        tuple: (session, error_message)
    """
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    try:
        # Print the model path to debug
        print(f"Loading model from: {model_path}")
        if not os.path.exists(model_path):
            return None, f"Model file not found at: {model_path}"
            
        sess = ort.InferenceSession(model_path, providers=providers)
        print(f"Using device: {ort.get_device()}")
        return sess, None
    except Exception as e:
        return None, f"Error creating inference session: {e}"


def get_classes_path(custom_path, default_path):
    """
    Get class names file path.
    
    Args:
        custom_path: Custom path to class names file
        default_path: Default path to class names file
        
    Returns:
        Path to a class names file
    """
    if not custom_path:
        return default_path
    
    # Treat as a file path
    if os.path.exists(custom_path):
        return custom_path
    
    return default_path


def load_class_names(class_names_path):
    """
    Load class names from a text file.

    Args:
        class_names_path: Path to a text file with class names (one per line)

    Returns:
        list: Class names or None if loading failed
    """
    if not class_names_path or not os.path.exists(class_names_path):
        return None

    try:
        with open(class_names_path, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(class_names)} class names from {class_names_path}")
        return class_names
    except Exception as e:
        print(f"Error loading class names: {e}")
        return None


def prepare_image(image, target_size=640):
    """
    Prepare image for inference by converting to PIL and resizing with padding.

    Args:
        image: Input image (PIL or numpy array)
        target_size: Target size for resizing (default: 640)

    Returns:
        tuple: (model_input, original_image, scale, padding)
    """
    # Convert to numpy array if PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Calculate scaling and padding
    height, width = image.shape[:2]
    scale = target_size / max(height, width)
    new_height = int(height * scale)
    new_width = int(width * scale)

    # Calculate padding
    y_offset = (target_size - new_height) // 2
    x_offset = (target_size - new_width) // 2

    # Create model input with padding
    model_input = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    model_input[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = cv2.resize(image, (new_width, new_height))

    return model_input, image, scale, (x_offset, y_offset)


def run_inference(session, image, target_size=640):
    """
    Run inference on the prepared image.

    Args:
        session: ONNX runtime session
        image: Prepared image array
        target_size: Target size used for padding

    Returns:
        tuple: (labels, boxes, scores)
    """
    # Convert BGR to RGB for model input
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Prepare input data
    im_data = np.ascontiguousarray(
        image_rgb.transpose(2, 0, 1),  # HWC to CHW format
        dtype=np.float32,
    )
    im_data = np.expand_dims(im_data, axis=0)  # Add batch dimension
    orig_size = np.array([[target_size, target_size]], dtype=np.int64)  # Use padded size

    # Get input name and run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(
        output_names=None,
        input_feed={input_name: im_data, "orig_target_sizes": orig_size},
    )

    return outputs


def count_objects(labels, scores, confidence_threshold, class_names):
    """
    Count detected objects by class.

    Args:
        labels: Detection labels
        scores: Detection confidence scores
        confidence_threshold: Minimum confidence threshold
        class_names: List of class names

    Returns:
        dict: Counts of objects by class
    """
    object_counts = {}
    for i, score_batch in enumerate(scores):
        for j, score in enumerate(score_batch):
            if score >= confidence_threshold:
                label = int(labels[i][j])
                class_name = (
                    class_names[label]
                    if class_names and label < len(class_names)
                    else f"Class {label}"
                )
                object_counts[class_name] = object_counts.get(class_name, 0) + 1

    return object_counts


def create_status_message(object_counts):
    """
    Create a status message with object counts.

    Args:
        object_counts: Dictionary of object counts by class

    Returns:
        str: Formatted status message
    """
    status_message = "Detection completed successfully\n\nObjects detected:"
    if object_counts:
        for class_name, count in object_counts.items():
            status_message += f"\n- {class_name}: {count}"
    else:
        status_message += "\n- No objects detected above confidence threshold"

    return status_message


def create_bar_data(object_counts):
    """
    Create data for the bar plot visualization.

    Args:
        object_counts: Dictionary of object counts by class

    Returns:
        DataFrame: Data for bar plot
    """
    if object_counts:
        # Sort by count in descending order
        sorted_counts = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
        class_names_list = [item[0] for item in sorted_counts]
        counts_list = [item[1] for item in sorted_counts]
        # Create a pandas DataFrame for the bar plot
        return pd.DataFrame({"Class": class_names_list, "Count": counts_list})
    else:
        return pd.DataFrame({"Class": ["No objects detected"], "Count": [0]})


def predict(image, model_path, class_names_path, confidence_threshold, image_size):
    """Main prediction function."""
    if image is None:
        return None, "Error: No image provided", None

    # Load model
    model_load_start = time.time()
    session, error = load_model(model_path)
    model_load_time = time.time() - model_load_start
    
    if error:
        return None, error, None

    # Load class names
    class_names = load_class_names(class_names_path)

    try:
        # Prepare image
        preprocess_start = time.time()
        model_input, original_image, scale, padding = prepare_image(image, image_size)
        preprocess_time = time.time() - preprocess_start

        # Run inference
        inference_start = time.time()
        outputs = run_inference(session, model_input, image_size)
        inference_time = time.time() - inference_start
        
        if not outputs or len(outputs) < 3:
            return None, "Error: Model output is invalid", None
            
        labels, boxes, scores = outputs

        # Draw detections
        postprocess_start = time.time()
        result_images = draw(
            [original_image],
            labels,
            boxes,
            scores,
            [scale],
            [padding],
            thrh=confidence_threshold,
            class_names=class_names,
        )

        # Count objects and create visualizations
        object_counts = count_objects(labels, scores, confidence_threshold, class_names)
        postprocess_time = time.time() - postprocess_start

        # Create status message with timing information
        status_message = create_status_message(object_counts)
        status_message += "\n\nLatency Information:"
        status_message += f"\n- Model Loading: {model_load_time*1000:.1f}ms"
        status_message += f"\n- Preprocessing: {preprocess_time*1000:.1f}ms"
        status_message += f"\n- Inference: {inference_time*1000:.1f}ms"
        status_message += f"\n- Postprocessing: {postprocess_time*1000:.1f}ms"
        status_message += f"\n- Total Time: {(model_load_time + preprocess_time + inference_time + postprocess_time)*1000:.1f}ms"
        
        bar_data = create_bar_data(object_counts)

        return result_images[0], status_message, bar_data
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during inference: {error_details}")
        return None, f"Error during inference: {str(e)}", None


def build_interface(model_path, class_names_path, example_images=None):
    """
    Build the Gradio interface components.

    Args:
        model_path: Path to the ONNX model
        class_names_path: Path to the class names file
        example_images: List of example image paths

    Returns:
        gr.Blocks: The Gradio demo interface
    """
    with gr.Blocks(title="DEIMKit Detection") as demo:
        gr.Markdown("# DEIMKit Detection")
        gr.Markdown("Configure the model and run inference on an image.")
        
        # Add model selection
        with gr.Accordion("Model Settings", open=False):
            with gr.Row():
                custom_model_path = gr.File(
                    label="Custom Model File (ONNX)",
                    file_types=[".onnx"],
                    file_count="single"
                )
                custom_classes_path = gr.File(
                    label="Custom Classes File (TXT)",
                    file_types=[".txt"],
                    file_count="single"
                )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image")
                
                with gr.Row():
                    confidence = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.4,
                        step=0.01,
                        label="Confidence Threshold",
                    )
                    
                    image_size = gr.Slider(
                        minimum=32,
                        maximum=1920,
                        value=640,
                        step=32,
                        label="Image Size",
                        info="Select image size for inference (larger = slower but potentially more accurate)"
                    )
                
                submit_btn = gr.Button("Run Inference", variant="primary")

            with gr.Column():
                output_image = gr.Image(type="pil", label="Detection Result")

                with gr.Row(equal_height=True):
                    output_message = gr.Textbox(label="Status")

                    count_plot = gr.BarPlot(
                        x="Class",
                        y="Count",
                        title="Object Counts",
                        tooltip=["Class", "Count"],
                        height=300,
                        orientation="h",
                        label_title="Object Counts",
                    )
        
        # Add examples component if example images are provided
        if example_images:
            gr.Examples(
                examples=example_images,
                inputs=input_image,
            )

        # Function to handle model path selection
        def get_model_path(custom_file, default_path):
            if custom_file is not None:
                return custom_file.name
            return default_path
            
        def get_classes_path(custom_file, default_path):
            if custom_file is not None:
                return custom_file.name
            return default_path

        # Set up the click event inside the Blocks context
        submit_btn.click(
            fn=lambda img, custom_model, custom_classes, conf, img_size: predict(
                img,
                get_model_path(custom_model, model_path),
                get_classes_path(custom_classes, class_names_path),
                conf,
                img_size
            ),
            inputs=[
                input_image,
                custom_model_path,
                custom_classes_path,
                confidence,
                image_size,
            ],
            outputs=[output_image, output_message, count_plot],
        )

        with gr.Row():
            with gr.Column():
                gr.HTML("<div style='text-align: center; margin: 0 auto;'>Created by <a href='https://dicksonneoh.com' target='_blank'>Dickson Neoh</a>.</div>")

        return demo


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DEIMKit Detection Demo')
    parser.add_argument(
        '--model', 
        type=str,
        default=MODEL_PATH,
        help='Path to ONNX model file'
    )
    parser.add_argument(
        '--classes', 
        type=str,
        default=CLASS_NAMES_PATH,
        help='Path to class names file'
    )
    parser.add_argument(
        '--examples',
        type=str,
        default=os.path.join(BASE_DIR, "examples"),
        help='Path to directory containing example images'
    )
    return parser.parse_args()


def launch_demo():
    """
    Launch the Gradio demo with model and class names paths from command line arguments.
    """
    args = parse_args()
    
    # Create examples directory if it doesn't exist
    examples_dir = args.examples
    if not os.path.exists(examples_dir):
        os.makedirs(examples_dir)
        print(f"Created examples directory at {examples_dir}")
    
    # Get list of example images
    example_images = []
    if os.path.exists(examples_dir):
        example_images = [
            os.path.join(examples_dir, f) 
            for f in os.listdir(examples_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        print(f"Found {len(example_images)} example images in {examples_dir}")
    
    demo = build_interface(args.model, args.classes, example_images)
    
    # Launch the demo without the examples parameter
    demo.launch(share=False, inbrowser=True)  # Set share=True if you want to create a shareable link


if __name__ == "__main__":
    launch_demo()
