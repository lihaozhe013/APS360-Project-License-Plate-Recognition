import os
import glob
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms

# Imports for models
import sys
# Add the src dir to the path so we can import from the subdirectories
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from aps360_lpr.train.src.bbox_model.bbox_cnn import BoundingBoxCNN
from aps360_lpr.train.src.recognition_model.crnn_class import CRNN, IDX2CHAR

def decode_predictions(predictions):
    """
    Decodes the raw output from the CRNN into a readable string.
    Removes consecutive duplicates and blank tokens.
    """
    _, max_indices = torch.max(predictions, 2)
    max_indices = max_indices.squeeze(1)

    decoded_str = []
    prev_idx = -1

    for idx in max_indices.tolist():
        # 0 is the blank index
        if idx != 0 and idx != prev_idx:
            decoded_str.append(IDX2CHAR[idx])
        prev_idx = idx

    return ''.join(decoded_str)

def get_bbox(image_rgb, bbox_model, device):
    """
    Given an RGB image, returns the 4 coordinate pairs [TL, TR, BL, BR].
    """
    orig_h, orig_w, _ = image_rgb.shape
    
    # Preprocess for bbox model
    resized_img = cv2.resize(image_rgb, (224, 224))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(resized_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = bbox_model(input_tensor).squeeze(0).cpu().numpy()
        
    # Introduce explicit bounding box padding/expansion margin (5%)
    # Prevents characters at the extreme edges from being cut down 
    # making the CRNN fail.
    PADDING_PERCENT = 0.05
    
    # Calculate geometric center of predicted points
    c_x = (output[0] + output[2] + output[4] + output[6]) / 4.0
    c_y = (output[1] + output[3] + output[5] + output[7]) / 4.0
    
    expanded_output = []
    for i in range(0, 8, 2):
        px = output[i]
        py = output[i+1]
        
        # Calculate vector dist from center to point
        dx = px - c_x
        dy = py - c_y
        
        # Scale the point outward
        px_padded = px + (dx * PADDING_PERCENT)
        py_padded = py + (dy * PADDING_PERCENT)
        
        expanded_output.extend([px_padded, py_padded])
        
    # Scale by original image dims
    TL = [expanded_output[0] * orig_w, expanded_output[1] * orig_h]
    TR = [expanded_output[2] * orig_w, expanded_output[3] * orig_h]
    BL = [expanded_output[4] * orig_w, expanded_output[5] * orig_h]
    BR = [expanded_output[6] * orig_w, expanded_output[7] * orig_h]
    
    return np.float32([TL, TR, BR, BL]) # Return as [TL, TR, BR, BL] for OpenCV warping

def read_text(warped_plate_bgr, crnn_model, device):
    """
    Given a cropped and warped BGR license plate image, return the text string.
    """
    # Convert BGR to Grayscale for CRNN
    gray = cv2.cvtColor(warped_plate_bgr, cv2.COLOR_BGR2GRAY)
    
    # The CRNN expects PIL Image or simple tensor
    image = Image.fromarray(gray)
    
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = crnn_model(image_tensor)
        
    text = decode_predictions(output)
    return text

def run_pipeline(val_dir, bbox_weights, crnn_weights):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on device: {device}")
    
    # 1. Load Bounding Box Model
    bbox_model = BoundingBoxCNN(pretrained=False)
    if os.path.exists(bbox_weights):
        bbox_model.load_state_dict(torch.load(bbox_weights, map_location=device))
    else:
        print(f"Warning: BBox model weights not found at {bbox_weights}. Predictions will be garbage.")
    bbox_model.to(device)
    bbox_model.eval()
    
    # 2. Load CRNN Model
    crnn_model = CRNN().to(device)
    if os.path.exists(crnn_weights):
        crnn_model.load_state_dict(torch.load(crnn_weights, map_location=device, weights_only=True))
    else:
        print(f"Warning: CRNN model weights not found at {crnn_weights}.")
    crnn_model.eval()
    
    image_paths = glob.glob(os.path.join(val_dir, "*.jpg")) + glob.glob(os.path.join(val_dir, "*.png"))
    
    if not image_paths:
        print(f"No images found in {val_dir}")
        return
        
    print(f"{'Image Name':<20} | {'Predicted Text':<15} | {'Match?'}")
    print("-" * 55)
    
    correct = 0
    total = len(image_paths)
    
    for img_path in image_paths:
        # Load image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 1. Run BBox Detection
        src_pts = get_bbox(img_rgb, bbox_model, device)
        
        # 2. Warp Plate Image
        # Standardize size to what CRNN expects (128x32)
        dst_pts = np.float32([[0, 0], [128, 0], [128, 32], [0, 32]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_plate = cv2.warpPerspective(img_bgr, M, (128, 32))
        
        # 3. Read Text
        predicted_text = read_text(warped_plate, crnn_model, device)
        
        # 4. Compare with Ground Truth
        actual_name = os.path.basename(img_path).split('.')[0]
        # In validation, actual_name matches the plate text exactly.
        
        is_match = (predicted_text == actual_name)
        if is_match:
            correct += 1
            
        print(f"{actual_name:<20} | {predicted_text:<15} | {'Yes' if is_match else 'No'}")
        
    print("-" * 55)
    print(f"Total: {total} | Correct: {correct} | Accuracy: {(correct/total)*100:.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="End-to-end Bounding Box + OCR Inference Pipeline")
    parser.add_argument("--val_dir", default="../../../../dataset/val", help="Directory containing validation images")
    parser.add_argument("--bbox_weights", default="bounding_box_model/weights/bbox_model.pth", help="Path to bbox model weights")
    parser.add_argument("--crnn_weights", default="crnn_plate_epoch_50.pth", help="Path to crnn model weights")
    
    args = parser.parse_args()
    
    run_pipeline(args.val_dir, args.bbox_weights, args.crnn_weights)
