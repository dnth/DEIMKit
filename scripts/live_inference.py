"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torchvision.transforms as T
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import cv2
import time
import colorsys


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
    num_classes = len(class_names) if class_names else 91  # Use length of class_names if available, otherwise default to COCO's 91 classes
    colors = generate_colors(num_classes)
    
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scr = scr[scr > thrh]

        ratio = ratios[i]
        pad_w, pad_h = paddings[i]

        for lbl, bb in zip(lab, box):
            # Get color for this class
            class_idx = int(lbl)
            color = colors[class_idx % len(colors)]
            
            # Convert RGB to hex for PIL
            hex_color = "#{:02x}{:02x}{:02x}".format(*color)
            
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
                label_text = f"{class_names[class_idx]} {scr[lab == lbl][0]:.2f}"
            else:
                label_text = f"Class {class_idx} {scr[lab == lbl][0]:.2f}"
            
            # Draw text background
            text_size = draw.textbbox((0, 0), label_text, font=None)
            text_width = text_size[2] - text_size[0]
            text_height = text_size[3] - text_size[1]
            
            # Draw text background rectangle
            draw.rectangle(
                [bb[0], bb[1] - text_height - 4, bb[0] + text_width + 4, bb[1]],
                fill=hex_color
            )
            
            # Draw text in white or black depending on color brightness
            brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
            text_color = "black" if brightness > 128 else "white"
            
            # Draw text
            draw.text((bb[0] + 2, bb[1] - text_height - 2), text=label_text, fill=text_color)

        result_images.append(im)
    return result_images


def process_image(sess, im_pil, class_names=None, input_size=640):
    # Resize image while preserving aspect ratio
    resized_im_pil, ratio, pad_w, pad_h = resize_with_aspect_ratio(im_pil, input_size)
    orig_size = torch.tensor([[resized_im_pil.size[1], resized_im_pil.size[0]]])

    transforms = T.Compose([
        T.ToTensor(),
    ])
    im_data = transforms(resized_im_pil).unsqueeze(0)

    output = sess.run(
        output_names=None,
        input_feed={'images': im_data.numpy(), "orig_target_sizes": orig_size.numpy()}
    )

    labels, boxes, scores = output

    result_images = draw(
        [im_pil], labels, boxes, scores,
        [ratio], [(pad_w, pad_h)],
        class_names=class_names
    )
    result_images[0].save('onnx_result.jpg')
    print("Image processing complete. Result saved as 'result.jpg'.")


def process_video(sess, video_path, class_names=None, input_size=640):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('onnx_result.mp4', fourcc, fps, (orig_w, orig_h))

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Resize frame while preserving aspect ratio
        resized_frame_pil, ratio, pad_w, pad_h = resize_with_aspect_ratio(frame_pil, input_size)
        orig_size = torch.tensor([[resized_frame_pil.size[1], resized_frame_pil.size[0]]])

        transforms = T.Compose([
            T.ToTensor(),
        ])
        im_data = transforms(resized_frame_pil).unsqueeze(0)

        output = sess.run(
            output_names=None,
            input_feed={'images': im_data.numpy(), "orig_target_sizes": orig_size.numpy()}
        )

        labels, boxes, scores = output

        # Draw detections on the original frame
        result_images = draw(
            [frame_pil], labels, boxes, scores,
            [ratio], [(pad_w, pad_h)],
            class_names=class_names
        )
        frame_with_detections = result_images[0]

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(frame_with_detections), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'result.mp4'.")


def process_webcam(sess, device_id=0, class_names=None, input_size=640):
    cap = cv2.VideoCapture(device_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open webcam device {device_id}")
        return
    
    print(f"Webcam opened successfully. Press 'q' to quit.")
    
    # Variables for FPS calculation
    prev_time = 0
    curr_time = 0
    fps = 0
    
    while True:
        # Calculate FPS
        curr_time = time.time()
        if curr_time - prev_time > 0:  # Avoid division by zero
            fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam")
            break
            
        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Resize frame while preserving aspect ratio
        resized_frame_pil, ratio, pad_w, pad_h = resize_with_aspect_ratio(frame_pil, input_size)
        orig_size = torch.tensor([[resized_frame_pil.size[1], resized_frame_pil.size[0]]])
        
        transforms = T.Compose([
            T.ToTensor(),
        ])
        im_data = transforms(resized_frame_pil).unsqueeze(0)
        
        output = sess.run(
            output_names=None,
            input_feed={'images': im_data.numpy(), "orig_target_sizes": orig_size.numpy()}
        )
        
        labels, boxes, scores = output
        
        # Draw detections on the original frame
        result_images = draw(
            [frame_pil], labels, boxes, scores,
            [ratio], [(pad_w, pad_h)],
            class_names=class_names
        )
        frame_with_detections = result_images[0]
        
        # Convert back to OpenCV image for display
        display_frame = cv2.cvtColor(np.array(frame_with_detections), cv2.COLOR_RGB2BGR)
        
        # Add FPS text to the top right corner with dark blue background
        fps_text = f"FPS: {fps:.1f}"
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = display_frame.shape[1] - text_size[0] - 10
        text_y = 30
        
        # Draw background rectangle
        cv2.rectangle(display_frame, 
                     (text_x - 5, text_y - text_size[1] - 5), 
                     (text_x + text_size[0] + 5, text_y + 5), 
                     (139, 0, 0), -1)  # Dark blue background (BGR format)
        
        # Draw text in white
        cv2.putText(display_frame, fps_text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow('Webcam Detection', display_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam processing stopped.")


def main(args):
    """Main function."""
    # Set up ONNX runtime session with specified provider
    providers = ['CPUExecutionProvider']
    if args.gpu:
        try:
            # Check if GPU providers are available
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print("Using CUDA for inference")
            elif 'TensorrtExecutionProvider' in available_providers:
                providers = ['TensorrtExecutionProvider', 'CPUExecutionProvider']
                print("Using TensorRT for inference")
            else:
                print("Warning: GPU requested but no GPU execution provider available. Falling back to CPU.")
        except Exception as e:
            print(f"Error initializing GPU providers: {e}")
            print("Falling back to CPU execution.")
            providers = ['CPUExecutionProvider']
    
    # Load the ONNX model with specified providers
    try:
        sess_options = ort.SessionOptions()
        if args.optimization_level is not None:
            # Convert integer to the proper GraphOptimizationLevel enum
            if args.optimization_level == 0:
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            elif args.optimization_level == 1:
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            elif args.optimization_level == 2:
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            elif args.optimization_level == 99:
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        sess = ort.InferenceSession(args.onnx, sess_options=sess_options, providers=providers)
        print(f"Using device: {ort.get_device()}")
    except Exception as e:
        print(f"Error creating inference session with providers {providers}: {e}")
        print("Attempting to fall back to CPU execution...")
        sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
        print(f"Using device: {ort.get_device()}")
    
    # Load class names if provided
    class_names = None
    if args.class_names:
        try:
            with open(args.class_names, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(class_names)} class names")
        except Exception as e:
            print(f"Error loading class names: {e}")
    
    # Get input size from args
    input_size = args.input_size
    
    if args.webcam:
        # Process webcam feed
        process_webcam(sess, args.device_id, class_names, input_size)
    else:
        input_path = args.input
        try:
            # Try to open the input as an image
            im_pil = Image.open(input_path).convert('RGB')
            process_image(sess, im_pil, class_names, input_size)
        except IOError:
            # Not an image, process as video
            process_video(sess, input_path, class_names, input_size)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, required=True, help='Path to the ONNX model file.')
    parser.add_argument('--input', type=str, help='Path to the input image or video file.')
    parser.add_argument('--webcam', action='store_true', help='Use webcam as input source')
    parser.add_argument('--device-id', type=int, default=0, help='Webcam device ID (default: 0)')
    parser.add_argument('--class-names', type=str, help='Path to a text file with class names (one per line)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')
    parser.add_argument('--optimization-level', type=int, choices=[0, 1, 2, 99], 
                        help='ONNX Runtime optimization level (0=disable, 1=basic, 2=extended, 99=all)')
    parser.add_argument('--input-size', type=int, default=640, 
                        help='Input image size for the model (default: 640)')
    args = parser.parse_args()
    
    if not args.webcam and not args.input:
        parser.error("Either --input or --webcam must be specified")
        
    main(args)
