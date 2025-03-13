import colorsys
import os

import gradio as gr
import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image, ImageDraw


# Use absolute paths instead of relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models/deim-blood-cell-detection_nano.onnx")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "models/classes.txt")

def resize_with_aspect_ratio(image, size, interpolation=Image.BILINEAR):
    """Resizes an image while maintaining aspect ratio and pads it."""
    original_width, original_height = image.size
    ratio = min(size / original_width, size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    image = image.resize((new_width, new_height), interpolation)

    # Create a new image with the desired size and paste the resized image onto it
    new_image = Image.new("RGB", (size, size))
    new_image.paste(image, ((size - new_width) // 2, (size - new_height) // 2))
    return new_image, ratio, (size - new_width) // 2, (size - new_height) // 2


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


def draw(images, labels, boxes, scores, ratios, paddings, thrh=0.4, class_names=None):
    result_images = []

    # Generate colors for classes
    num_classes = (
        len(class_names) if class_names else 91
    )  # Use length of class_names if available, otherwise default to COCO's 91 classes
    colors = generate_colors(num_classes)

    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        
        # Get indices of scores above threshold
        valid_indices = np.where(scr > thrh)[0]
        
        # Filter using these indices
        valid_labels = labels[i][valid_indices]
        valid_boxes = boxes[i][valid_indices]
        valid_scores = scr[valid_indices]

        for j, (lbl, bb, score) in enumerate(zip(valid_labels, valid_boxes, valid_scores)):
            # Get color for this class
            class_idx = int(lbl)
            color = colors[class_idx % len(colors)]

            # Convert RGB to hex for PIL
            hex_color = "#{:02x}{:02x}{:02x}".format(*color)

            ratio = ratios[i]
            pad_w, pad_h = paddings[i]
            
            # Adjust bounding boxes according to the resizing and padding
            bb = [
                (bb[0] - pad_w) / ratio,
                (bb[1] - pad_h) / ratio,
                (bb[2] - pad_w) / ratio,
                (bb[3] - pad_h) / ratio,
            ]

            # Draw rectangle with class-specific color
            draw.rectangle(bb, outline=hex_color, width=3)

            # Use class name if available, otherwise use class index
            if class_names and class_idx < len(class_names):
                label_text = f"{class_names[class_idx]} {score:.2f}"
            else:
                label_text = f"Class {class_idx} {score:.2f}"

            # Draw text background
            text_size = draw.textbbox((0, 0), label_text, font=None)
            text_width = text_size[2] - text_size[0]
            text_height = text_size[3] - text_size[1]

            # Draw text background rectangle
            draw.rectangle(
                [bb[0], bb[1] - text_height - 4, bb[0] + text_width + 4, bb[1]],
                fill=hex_color,
            )

            # Draw text in white or black depending on color brightness
            brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
            text_color = "black" if brightness > 128 else "white"

            # Draw text
            draw.text(
                (bb[0] + 2, bb[1] - text_height - 2), text=label_text, fill=text_color
            )

        result_images.append(im)
    return result_images


def load_model(model_path):
    """
    Load an ONNX model for inference.

    Args:
        model_path: Path to the ONNX model file

    Returns:
        tuple: (session, error_message)
    """
    providers = ["CPUExecutionProvider"]

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


def prepare_image(image):
    """
    Prepare image for inference by converting to PIL and resizing.

    Args:
        image: Input image (PIL or numpy array)

    Returns:
        tuple: (resized_image, original_image, ratio, padding)
    """
    # Convert to PIL image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")

    # Resize image while preserving aspect ratio
    resized_image, ratio, pad_w, pad_h = resize_with_aspect_ratio(image, 320)

    return resized_image, image, ratio, (pad_w, pad_h)


def run_inference(session, image):
    """
    Run inference on the prepared image.

    Args:
        session: ONNX runtime session
        image: Prepared PIL image

    Returns:
        tuple: (labels, boxes, scores)
    """
    # Check if image is None
    if image is None:
        raise ValueError("Input image is None")
        
    # Get original image dimensions
    orig_height, orig_width = image.size[1], image.size[0]
    # Convert to int64 as expected by the model
    orig_size = np.array([[orig_height, orig_width]], dtype=np.int64)

    # Convert PIL image to numpy array and normalize to 0-1 range
    im_data = np.array(image, dtype=np.float32) / 255.0
    # Transpose from HWC to CHW format
    im_data = im_data.transpose(2, 0, 1)
    # Add batch dimension
    im_data = np.expand_dims(im_data, axis=0)

    output = session.run(
        output_names=None,
        input_feed={"images": im_data, "orig_target_sizes": orig_size},
    )

    return output  # labels, boxes, scores


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


def predict(image, model_path, class_names_path, confidence_threshold):
    """
    Main prediction function that orchestrates the detection pipeline.

    Args:
        image: Input image
        model_path: Path to ONNX model
        class_names_path: Path to class names file or list of class names
        confidence_threshold: Detection confidence threshold

    Returns:
        tuple: (result_image, status_message, bar_data)
    """
    # Check if image is None
    if image is None:
        return None, "Error: No image provided", None
        
    # Load model
    session, error = load_model(model_path)
    if error:
        return None, error, None

    # Load class names
    class_names = load_class_names(class_names_path)
    
    # Debug print to verify class names are loaded correctly
    print(f"Class names for detection: {class_names}")

    try:
        # Prepare image
        resized_image, original_image, ratio, padding = prepare_image(image)

        # Run inference
        output = run_inference(session, resized_image)
        
        # Check if output is valid
        if not output or len(output) < 3:
            return None, "Error: Model output is invalid", None
            
        labels, boxes, scores = output

        # Draw detections on the original image
        result_images = draw(
            [original_image],
            labels,
            boxes,
            scores,
            [ratio],
            [padding],
            thrh=confidence_threshold,
            class_names=class_names,
        )

        # Count objects by class
        object_counts = count_objects(labels, scores, confidence_threshold, class_names)
        
        # Debug print to verify object counts
        print(f"Object counts: {object_counts}")

        # Create status message
        status_message = create_status_message(object_counts)

        # Create bar plot data
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
        with gr.Accordion("Model Settings", open=True):
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
                confidence = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.4,
                    step=0.01,
                    label="Confidence Threshold",
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
            fn=lambda img, custom_model, custom_classes, conf: predict(
                img,
                get_model_path(custom_model, model_path),
                get_classes_path(custom_classes, class_names_path),
                conf
            ),
            inputs=[
                input_image,
                custom_model_path,
                custom_classes_path,
                confidence,
            ],
            outputs=[output_image, output_message, count_plot],
        )

        with gr.Row():
            with gr.Column():
                gr.HTML("<div style='text-align: center; margin: 0 auto;'>Created by <a href='https://dicksonneoh.com' target='_blank'>Dickson Neoh</a>.</div>")

        return demo


def launch_demo():
    """
    Launch the Gradio demo with hardcoded model and class names paths.
    """
    # Create examples directory if it doesn't exist
    examples_dir = os.path.join(BASE_DIR, "examples")
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
        print(f"Found {len(example_images)} example images")
    
    demo = build_interface(MODEL_PATH, CLASS_NAMES_PATH, example_images)
    
    # Launch the demo without the examples parameter
    demo.launch(share=False, inbrowser=True)  # Set share=True if you want to create a shareable link


if __name__ == "__main__":
    launch_demo()
