import colorsys
import time
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

ort.preload_dlls()


def generate_colors(num_classes):
    """Generate distinct colors for visualization."""
    hsv_tuples = [(x / num_classes, 0.8, 0.9) for x in range(num_classes)]
    colors = []
    for hsv in hsv_tuples:
        rgb = colorsys.hsv_to_rgb(*hsv)
        colors.append(tuple(int(255 * x) for x in rgb))
    return colors


def draw_boxes(
    image: np.ndarray,
    labels: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.3,
    class_names: Optional[list[str]] = None,
) -> np.ndarray:
    """Draw bounding boxes on the image with detection results.

    Args:
        image: Input image array in BGR format
        labels: Array of class labels
        boxes: Array of bounding box coordinates [x1, y1, x2, y2]
        scores: Array of confidence scores
        threshold: Minimum confidence threshold for displaying detections
        class_names: Optional list of class names for labels

    Returns:
        np.ndarray: Image with drawn bounding boxes in BGR format
    """
    # Generate colors for classes
    num_classes = len(class_names) if class_names else 91
    colors = generate_colors(num_classes)

    # Filter detections by threshold
    valid_indices = scores > threshold
    labels = labels[valid_indices]
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]

    for j, (lbl, box, score) in enumerate(zip(labels, boxes, scores)):
        # Get color for this class
        class_idx = int(lbl)
        color = colors[class_idx % len(colors)]

        # Use box coordinates directly
        box_coords = [
            int(box[0]),  # x1
            int(box[1]),  # y1
            int(box[2]),  # x2
            int(box[3]),  # y2
        ]

        # Draw rectangle
        cv2.rectangle(
            image,
            (box_coords[0], box_coords[1]),
            (box_coords[2], box_coords[3]),
            color,
            2,
        )

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
            image,
            (box_coords[0], box_coords[1] - text_height - 4),
            (box_coords[0] + text_width + 4, box_coords[1]),
            color,
            -1,  # Filled rectangle
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
            2,
        )

    return image


def load_model(model_path: str, provider: str = "cpu") -> ort.InferenceSession:
    """Initialize and load the ONNX model with specified provider.

    Args:
        model_path: Path to the ONNX model file
        provider: Provider to use for inference ("cpu", "cuda", or "tensorrt")

    Returns:
        ort.InferenceSession: Initialized ONNX Runtime session

    Raises:
        RuntimeError: If model loading fails with specified provider
    """
    
    if provider == "cpu":
        providers = ["CPUExecutionProvider"]
    elif provider == "cuda":
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            ),
            "CPUExecutionProvider",
        ]
    elif provider == "tensorrt":
        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    "trt_fp16_enable": False,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "./trt_cache",
                    "trt_timing_cache_enable": True,
                },
            ),
            "CPUExecutionProvider",
        ]
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    try:
        print(f"Loading ONNX model with providers: {providers}...")
        session = ort.InferenceSession(model_path, providers=providers)
        print(f"Using provider: {session.get_providers()[0]}")
        return session
    except Exception as e:
        print(f"Error creating inference session with providers {providers}: {e}")
        print("Attempting to fall back to CPU execution...")
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        return session


def run_inference(
    model_path: str,
    input_source: str | int,  # Can be image path, video path, or camera index
    class_names_path: str | None = None,
    threshold: float = 0.3,
    provider: str = "cpu",
    inference_size: int = 640,  # renamed from video_width
) -> None:
    """Run object detection on images or video streams.
    
    Args:
        model_path: Path to the ONNX model file
        input_source: Path to image/video file or camera index (usually 0 for webcam)
        class_names_path: Optional path to class names file
        threshold: Detection confidence threshold
        provider: ONNXRuntime provider ("cpu", "cuda", "tensorrt")
        inference_size: Size for inference processing (default: 640). Larger dimension will be scaled to this size while preserving aspect ratio
    """
    # Load model and class names
    session = load_model(model_path, provider)
    class_names = None
    if class_names_path:
        try:
            with open(class_names_path, "r") as f:
                class_names = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(class_names)} class names")
        except Exception as e:
            print(f"Error loading class names: {e}")

    # Determine if input is image file or video source
    if isinstance(input_source, str) and any(
        input_source.lower().endswith(ext) 
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']
    ):
        # Handle single image
        image = cv2.imread(input_source)
        if image is None:
            raise RuntimeError(f"Failed to load image: {input_source}")
            
        result = process_frame(
            image,
            session,
            class_names,
            threshold,
            inference_size
        )
        
        # Save and display result
        output_path = "detection_result.jpg"
        cv2.imwrite(output_path, result)
        print(f"Detection complete. Result saved to {output_path}")
        
        cv2.imshow("Detection Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        # Handle video/webcam
        cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {input_source}")

        # Configure video capture
        if isinstance(input_source, int):  # Webcam settings
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
            cap.set(cv2.CAP_PROP_FPS, 100)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, inference_size)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(inference_size * 9 / 16))

        prev_time = time.time()
        fps_display = 0
        show_overlay = True

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Calculate FPS
                current_time = time.time()
                fps_display = 1 / (current_time - prev_time) if current_time - prev_time > 0 else 0
                prev_time = current_time

                # Process frame
                result = process_frame(
                    frame,
                    session,
                    class_names,
                    threshold,
                    inference_size
                )

                # Add overlay
                result = draw_text_overlay(
                    result,
                    fps_display,
                    session.get_providers()[0],
                    inference_size,
                    show_overlay
                )

                # Display result
                cv2.imshow("Detection", result)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("t"):
                    show_overlay = not show_overlay

        finally:
            cap.release()
            cv2.destroyAllWindows()


def process_frame(
    frame: np.ndarray,
    session: ort.InferenceSession,
    class_names: list[str] | None,
    threshold: float,
    target_width: int,
) -> np.ndarray:
    """Process a single frame through the object detection model.
    
    Args:
        frame: Input frame in BGR format
        session: ONNX Runtime inference session
        class_names: Optional list of class names
        threshold: Detection confidence threshold
        target_width: Target width for processing
        
    Returns:
        np.ndarray: Processed frame with detections
    """
    # Calculate scaling and padding
    height, width = frame.shape[:2]
    scale = target_width / max(height, width)
    new_height = int(height * scale)
    new_width = int(width * scale)

    # Calculate padding
    y_offset = (target_width - new_height) // 2
    x_offset = (target_width - new_width) // 2

    # Create model input with padding
    model_input = np.zeros((target_width, target_width, 3), dtype=np.uint8)
    model_input[
        y_offset : y_offset + new_height,
        x_offset : x_offset + new_width
    ] = cv2.resize(frame, (new_width, new_height))

    # Prepare input data
    im_data = np.ascontiguousarray(
        model_input.transpose(2, 0, 1),
        dtype=np.float32,
    )
    im_data = np.expand_dims(im_data, axis=0)
    orig_size = np.array([[target_width, target_width]], dtype=np.int64)

    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(
        output_names=None,
        input_feed={input_name: im_data, "orig_target_sizes": orig_size},
    )

    # Process outputs
    labels, boxes, scores = outputs
    
    # Scale boxes back to original frame size
    boxes = boxes[0]  # Remove batch dimension
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - x_offset) / scale  # x coordinates
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - y_offset) / scale  # y coordinates

    # Draw detections
    return draw_boxes(
        frame,
        labels[0],
        boxes,
        scores[0],
        threshold=threshold,
        class_names=class_names,
    )


def draw_text_overlay(
    image: np.ndarray,
    fps: float,
    provider: str,
    video_width: int,
    show_overlay: bool = True,
) -> np.ndarray:
    """Draw text overlays (FPS, width, provider) on the detection frame.
    
    Args:
        image: Input image array in BGR format
        fps: Current FPS value to display
        provider: Provider name being used
        video_width: Current video width in pixels
        show_overlay: Whether to show the text overlay
        
    Returns:
        np.ndarray: Image with text overlays
    """
    if not show_overlay:
        return image

    # Add video width display at top left with dark green background
    width_text = f"Width: {int(video_width)}px"
    text_size = cv2.getTextSize(width_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]

    # Draw dark green background rectangle
    cv2.rectangle(
        image,
        (5, 5),  # Slight padding from corner
        (text_size[0] + 15, 35),  # Add padding around text
        (0, 100, 0),  # Dark green in BGR
        -1,  # Filled rectangle
    )

    # Draw width text
    cv2.putText(
        image,
        width_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),  # White text
        2,
    )

    # Add FPS display
    fps_text = f"FPS: {fps:.1f}"
    text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = image.shape[1] - text_size[0] - 10
    text_y = 30

    # Draw FPS background rectangle
    cv2.rectangle(
        image,
        (text_x - 5, text_y - text_size[1] - 5),
        (text_x + text_size[0] + 5, text_y + 5),
        (139, 0, 0),
        -1,
    )

    # Draw FPS text
    cv2.putText(
        image,
        fps_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    # Add provider display
    provider_text = f"Provider: {provider}"
    text_size = cv2.getTextSize(provider_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = image.shape[0] - 20

    # Draw provider background rectangle
    cv2.rectangle(
        image,
        (text_x - 5, text_y - text_size[1] - 5),
        (text_x + text_size[0] + 5, text_y + 5),
        (0, 0, 139),
        -1,
    )

    # Draw provider text
    cv2.putText(
        image,
        provider_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    return image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple ONNX object detection")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to ONNX model file"
    )
    parser.add_argument("--image", type=str, help="Path to input image (optional)")
    parser.add_argument("--webcam", action="store_true", help="Use webcam input")
    parser.add_argument(
        "--inference-size",
        type=int,
        default=640,
        help="Size for inference processing (default: 640). Larger dimension will be scaled to this size while preserving aspect ratio",
    )
    parser.add_argument(
        "--classes", type=str, help="Path to class names file (optional)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["cpu", "cuda", "tensorrt"],
        default="cpu",
        help="ONNXRuntime provider to use for inference",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Detection confidence threshold (default: 0.3)",
    )
    parser.add_argument("--video", type=str, help="Path to input video file (optional)")

    args = parser.parse_args()

    if args.webcam:
        run_inference(
            args.model, 0, args.classes, args.threshold, args.provider, args.inference_size
        )
    elif args.video:
        run_inference(
            args.model,
            args.video,
            args.classes,
            args.threshold,
            args.provider,
            args.inference_size,
        )
    elif args.image:
        run_inference(
            args.model, args.image, args.classes, args.threshold, args.provider, args.inference_size
        )
    else:
        parser.error("Either --image, --video, or --webcam must be specified")
