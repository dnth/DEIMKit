import colorsys
import time

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
    image, labels, boxes, scores, ratio, padding, threshold=0.3, class_names=None
):
    """Draw bounding boxes on the image."""
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


def run_inference(
    model_path, image_path, class_names_path=None, threshold=0.3, provider="cpu"
):
    # Set up providers based on selection
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
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "./trt_cache",
                    "trt_timing_cache_enable": True,
                },
            ),
            "CPUExecutionProvider",
        ]

    try:
        print(f"Loading ONNX model with providers: {providers}...")
        session = ort.InferenceSession(model_path, providers=providers)
        print(f"Using provider: {session.get_providers()[0]}")
    except Exception as e:
        print(f"Error creating inference session with providers {providers}: {e}")
        print("Attempting to fall back to CPU execution...")
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

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

    im_data = np.ascontiguousarray(
        image.transpose(2, 0, 1),  # HWC to CHW format
        dtype=np.float32,
    )
    im_data = np.expand_dims(im_data, axis=0)  # Add batch dimension
    orig_size = np.array([[image.shape[0], image.shape[1]]], dtype=np.int64)

    print(f"Image frame shape: {image.shape}")
    print(f"Processed input shape: {im_data.shape}")

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
        threshold=threshold,
        class_names=class_names,
    )

    # Save and show result
    output_path = "detection_result.jpg"
    result_bgr = cv2.cvtColor(
        result_image, cv2.COLOR_RGB2BGR
    )  # Convert back to BGR for OpenCV
    cv2.imwrite(output_path, result_bgr)
    print(f"Detection complete. Result saved to {output_path}")

    # Display the result
    cv2.imshow("Detection Result", result_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result_image


def run_inference_webcam(
    model_path, class_names_path=None, provider="cpu", threshold=0.3, video_width=640
):
    """Run real-time object detection on webcam feed."""
    # Set up providers based on selection
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

    try:
        print(f"Loading ONNX model with providers: {providers}...")
        session = ort.InferenceSession(model_path, providers=providers)
        print(f"Using provider: {session.get_providers()[0]}")
    except Exception as e:
        print(f"Error creating inference session with providers {providers}: {e}")
        print("Attempting to fall back to CPU execution...")
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # Update FPS calculation variables
    prev_time = time.time()
    fps_display = 0

    # Load class names if provided
    class_names = None
    if class_names_path:
        try:
            with open(class_names_path, "r") as f:
                class_names = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(class_names)} class names")
        except Exception as e:
            print(f"Error loading class names: {e}")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam")

    # Set camera to maximum possible FPS
    cap.set(
        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G")
    )  # Use MJPG format for higher FPS
    cap.set(
        cv2.CAP_PROP_FPS, 1000
    )  # Request very high FPS - will default to max supported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(video_width * 9 / 16))  # 16:9 aspect ratio

    # Print actual camera properties
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(
        f"Camera settings - FPS: {actual_fps}, Resolution: {actual_width}x{actual_height}"
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Calculate FPS
            current_time = time.time()
            if current_time - prev_time > 0:  # Avoid division by zero
                fps_display = 1 / (current_time - prev_time)
            prev_time = current_time

            # Calculate scaling and padding
            height, width = frame.shape[:2]
            scale = 640.0 / max(height, width)
            new_height = int(height * scale)
            new_width = int(width * scale)

            # Calculate padding
            y_offset = (640 - new_height) // 2
            x_offset = (640 - new_width) // 2

            # Create model input with padding
            model_input = np.zeros((640, 640, 3), dtype=np.uint8)
            model_input[
                y_offset : y_offset + new_height, x_offset : x_offset + new_width
            ] = cv2.resize(frame, (new_width, new_height))

            # Convert BGR to RGB for model input
            image = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)

            # Prepare input data
            im_data = np.ascontiguousarray(
                image.transpose(2, 0, 1),
                dtype=np.float32,
            )
            im_data = np.expand_dims(im_data, axis=0)
            orig_size = np.array([[640, 640]], dtype=np.int64)  # Use padded size

            # Get input name and run inference
            input_name = session.get_inputs()[0].name
            outputs = session.run(
                output_names=None,
                input_feed={input_name: im_data, "orig_target_sizes": orig_size},
            )

            # Process outputs
            labels, boxes, scores = outputs

            # Scale boxes from padded 640x640 to original frame size
            boxes = boxes[0]  # Remove batch dimension
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - x_offset) / scale  # x coordinates
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - y_offset) / scale  # y coordinates

            # Draw bounding boxes on the original frame
            result_image = draw_boxes(
                frame,  # Use original frame
                labels[0],
                boxes,
                scores[0],
                1.0,  # No additional scaling needed
                (0, 0),  # No additional padding needed
                threshold=threshold,
                class_names=class_names,
            )

            # No need to convert back to BGR since we're using the original frame
            result_bgr = result_image

            # Add video width display at top left with dark green background
            width_text = f"Width: {int(actual_width)}px"
            text_size = cv2.getTextSize(width_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            
            # Draw dark green background rectangle
            cv2.rectangle(
                result_bgr,
                (5, 5),  # Slight padding from corner
                (text_size[0] + 15, 35),  # Add padding around text
                (0, 100, 0),  # Dark green in BGR
                -1,  # Filled rectangle
            )

            # Draw text
            cv2.putText(
                result_bgr,
                width_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),  # White text
                2,
            )

            # Add FPS display (existing code)
            fps_text = f"FPS: {fps_display:.1f}"
            text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = result_bgr.shape[1] - text_size[0] - 10
            text_y = 30

            # Draw FPS background rectangle
            cv2.rectangle(
                result_bgr,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                (139, 0, 0),
                -1,
            )

            # Draw FPS text
            cv2.putText(
                result_bgr,
                fps_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # Add provider display
            provider_text = f"Provider: {session.get_providers()[0]}"
            text_size = cv2.getTextSize(
                provider_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )[0]
            text_x = (result_bgr.shape[1] - text_size[0]) // 2
            text_y = result_bgr.shape[0] - 20

            # Draw provider background rectangle
            cv2.rectangle(
                result_bgr,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                (0, 0, 139),
                -1,
            )

            # Draw provider text
            cv2.putText(
                result_bgr,
                provider_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # Display the result
            cv2.imshow("Webcam Detection", result_bgr)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_inference_video(
    model_path,
    video_path,
    class_names_path=None,
    provider="cpu",
    threshold=0.3,
    video_width=640,
):
    """Run object detection on a video file."""
    # Set up providers (same as webcam function)
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

    # Initialize model session
    try:
        print(f"Loading ONNX model with providers: {providers}...")
        session = ort.InferenceSession(model_path, providers=providers)
        print(f"Using provider: {session.get_providers()[0]}")
    except Exception as e:
        print(f"Error creating inference session with providers {providers}: {e}")
        print("Attempting to fall back to CPU execution...")
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # Load class names
    class_names = None
    if class_names_path:
        try:
            with open(class_names_path, "r") as f:
                class_names = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(class_names)} class names")
        except Exception as e:
            print(f"Error loading class names: {e}")

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate output dimensions based on video_width
    scale = video_width / frame_width
    output_width = video_width
    output_height = int(frame_height * scale)

    # Create video writer with new dimensions
    output_path = "detection_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

    # Initialize FPS calculation
    prev_time = time.time()
    fps_display = 0
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 10 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Processing: {progress:.1f}% complete", end="\r")

            # Calculate FPS
            current_time = time.time()
            if current_time - prev_time > 0:
                fps_display = 1 / (current_time - prev_time)
            prev_time = current_time

            # Calculate scaling and padding using video_width parameter
            height, width = frame.shape[:2]
            scale = video_width / max(height, width)
            new_height = int(height * scale)
            new_width = int(width * scale)

            # Calculate padding
            y_offset = (video_width - new_height) // 2
            x_offset = (video_width - new_width) // 2

            # Create model input with padding using video_width
            model_input = np.zeros((video_width, video_width, 3), dtype=np.uint8)
            model_input[
                y_offset : y_offset + new_height, x_offset : x_offset + new_width
            ] = cv2.resize(frame, (new_width, new_height))

            # Convert BGR to RGB for model input
            image = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)

            # Prepare input data
            im_data = np.ascontiguousarray(
                image.transpose(2, 0, 1),
                dtype=np.float32,
            )
            im_data = np.expand_dims(im_data, axis=0)
            orig_size = np.array(
                [[video_width, video_width]], dtype=np.int64
            )  # Use padded size

            # Run inference
            input_name = session.get_inputs()[0].name
            outputs = session.run(
                output_names=None,
                input_feed={input_name: im_data, "orig_target_sizes": orig_size},
            )

            # Process outputs
            labels, boxes, scores = outputs

            # Scale boxes from padded 640x640 to original frame size
            boxes = boxes[0]  # Remove batch dimension
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - x_offset) / scale  # x coordinates
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - y_offset) / scale  # y coordinates

            # Draw bounding boxes on the original frame
            result_image = draw_boxes(
                frame,  # Use original frame
                labels[0],
                boxes,
                scores[0],
                1.0,  # No additional scaling needed
                (0, 0),  # No additional padding needed
                threshold=threshold,
                class_names=class_names,
            )

            # Before writing the frame, resize it
            result_image = cv2.resize(result_image, (output_width, output_height))
            out.write(result_image)

            # Add video width display at top left with dark green background
            width_text = f"Width: {output_width}px"
            text_size = cv2.getTextSize(width_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            
            # Draw dark green background rectangle
            cv2.rectangle(
                result_image,
                (5, 5),  # Slight padding from corner
                (text_size[0] + 15, 35),  # Add padding around text
                (0, 100, 0),  # Dark green in BGR
                -1,  # Filled rectangle
            )

            # Draw text
            cv2.putText(
                result_image,
                width_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),  # White text
                2,
            )

            # Add FPS counter and provider info (existing code)
            fps_text = f"FPS: {fps_display:.1f}"
            text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = result_image.shape[1] - text_size[0] - 10
            text_y = 30

            # Draw FPS background rectangle
            cv2.rectangle(
                result_image,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                (139, 0, 0),
                -1,
            )

            # Draw FPS text
            cv2.putText(
                result_image,
                fps_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # Add provider display at bottom (matching webcam style)
            provider_text = f"Provider: {session.get_providers()[0]}"
            text_size = cv2.getTextSize(
                provider_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )[0]
            text_x = (result_image.shape[1] - text_size[0]) // 2
            text_y = result_image.shape[0] - 20

            # Draw provider background rectangle
            cv2.rectangle(
                result_image,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                (139, 0, 0),
                -1,
            )

            # Draw provider text
            cv2.putText(
                result_image,
                provider_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # Display frame (optional)
            cv2.imshow("Video Detection", result_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"\nVideo processing complete. Output saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple ONNX object detection")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to ONNX model file"
    )
    parser.add_argument("--image", type=str, help="Path to input image (optional)")
    parser.add_argument("--webcam", action="store_true", help="Use webcam input")
    parser.add_argument(
        "--video-width",
        type=int,
        default=640,
        help="Width of the video input in pixels (default: 640). Height will be adjusted to maintain aspect ratio",
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
        run_inference_webcam(
            args.model, args.classes, args.provider, args.threshold, args.video_width
        )
    elif args.video:
        run_inference_video(
            args.model,
            args.video,
            args.classes,
            args.provider,
            args.threshold,
            args.video_width,
        )
    elif args.image:
        run_inference(
            args.model, args.image, args.classes, args.threshold, args.provider
        )
    else:
        parser.error("Either --image, --video, or --webcam must be specified")
