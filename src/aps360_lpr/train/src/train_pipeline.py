import os
import glob
import cv2
import numpy as np
import torch
import shutil
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms

# Imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from aps360_lpr.train.src.bbox_model.bbox_cnn import BoundingBoxCNN

def get_bbox(image_rgb, bbox_model, device):
    """
    Given an RGB image, returns the 4 coordinate pairs [TL, TR, BL, BR].
    """
    orig_h, orig_w, _ = image_rgb.shape
    
    # Preprocess
    scale_ratio = 800.0 / orig_w
    low_res_w = int(orig_w * scale_ratio)
    low_res_h = int(orig_h * scale_ratio)
    low_res_img = cv2.resize(image_rgb, (low_res_w, low_res_h), interpolation=cv2.INTER_AREA)
    
    resized_img = cv2.resize(low_res_img, (224, 224), interpolation=cv2.INTER_AREA)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(resized_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = bbox_model(input_tensor).squeeze(0).cpu().numpy()
        
    PADDING_PERCENT = 0.05
    c_x = (output[0] + output[2] + output[4] + output[6]) / 4.0
    c_y = (output[1] + output[3] + output[5] + output[7]) / 4.0
    
    expanded_output = []
    for i in range(0, 8, 2):
        px = output[i]
        py = output[i+1]
        
        dx = px - c_x
        dy = py - c_y
        
        px_padded = px + (dx * PADDING_PERCENT)
        py_padded = py + (dy * PADDING_PERCENT)
        expanded_output.extend([px_padded, py_padded])
        
    TL = [expanded_output[0] * orig_w, expanded_output[1] * orig_h]
    TR = [expanded_output[2] * orig_w, expanded_output[3] * orig_h]
    BL = [expanded_output[4] * orig_w, expanded_output[5] * orig_h]
    BR = [expanded_output[6] * orig_w, expanded_output[7] * orig_h]
    
    return np.float32([TL, TR, BR, BL])

def preprocess_and_crop_dataset(input_dir, output_dir, bbox_weights):
    """
    Reads the embedded full images, runs BBox model to get coordinates, 
    crops the license plate to 128x32, and saves them ready for CRNN.
    """
    import random
    
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for BBox cropping: {device}")
    
    bbox_model = BoundingBoxCNN(pretrained=False)
    bbox_model.load_state_dict(torch.load(bbox_weights, map_location=device))
    bbox_model.eval()
    bbox_model.to(device)
    
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))
    
    # Shuffle for train/val split
    random.shuffle(image_paths)
    
    print(f"Found {len(image_paths)} full images. Cropping plates...")
    
    for i, img_path in enumerate(tqdm(image_paths)):
        # Format: embedded_x1_y1_..._x4_y4_PLATESTRING_idx.jpg
        # Extract the plate string from filename
        filename = os.path.basename(img_path)
        parts = filename.split('_')
        
        try:
            # We skip "embedded" (0) and 8 coordinates (1-8), taking the string at index 9
            plate_string = parts[9]
        except IndexError:
            # Maybe the format failed
            continue
            
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Predict coordinates for bounding box locally via the neural network
        src_pts = get_bbox(img_rgb, bbox_model, device)
        
        # Warp the predicted crop to a 128x32 rectangle
        dst_pts = np.float32([[0, 0], [128, 0], [128, 32], [0, 32]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_plate = cv2.warpPerspective(img_bgr, M, (128, 32))
        
        # Save it for CRNN layout under the strict naming convention: PLATESTRING_idx.jpg
        out_filename = f"{plate_string}_{parts[-1]}"
        
        # Split 95% train, 5% val
        if i < int(len(image_paths) * 0.95):
            cv2.imwrite(os.path.join(train_dir, out_filename), warped_plate)
        else:
            cv2.imwrite(os.path.join(val_dir, out_filename), warped_plate)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedded_dir", default="../../../../dataset/background_embedder_out_")
    parser.add_argument("--crnn_train_dir", default="../../../../dataset/crnn_training_data")
    parser.add_argument("--bbox_weights", default="bbox_model/weights/bbox_model.pth")
    args = parser.parse_args()

    # 1. Produce Cropped dataset using inference from bbox network
    print("--- 1. Generating Cropped Dataset for CRNN ---")
    preprocess_and_crop_dataset(args.embedded_dir, args.crnn_train_dir, args.bbox_weights)
    
    # 2. Delegate to your CRNN routine
    print("\n--- 2. Starting CRNN Training Pipeline ---")
    # Because your crnn training routine sits elsewhere, we can directly invoke it
    # We call standard python modules mapped directly
    # Adjust variables like 'epochs' or 'batch_size' by modifying recognition_train.py directly
    
    import subprocess
    crnn_train_script = os.path.join(os.path.dirname(__file__), "recognition_model", "recognition_train.py")
    
    # Execute the crnn training script with the newly prepared folder
    subprocess.run(["python", crnn_train_script, "--data_dir", args.crnn_train_dir])


if __name__ == "__main__":
    main()
