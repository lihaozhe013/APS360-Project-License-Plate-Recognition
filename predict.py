# uv run predict.py "path/to/my_random_image.jpg"
import argparse
import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path

# Automatically add the models directory to the path so we can import them
project_root = Path(__file__).parent.resolve()
inference_dir = project_root / "src" / "aps360_lpr" / "train" / "src"
sys.path.append(str(inference_dir))

# Import the existing pipeline tools
from aps360_lpr.train.src.inference_pipeline import get_bbox, read_text
from aps360_lpr.train.src.bbox_model.bbox_cnn import BoundingBoxCNN
from aps360_lpr.train.src.recognition_model.crnn_class import CRNN

def predict_single_image(image_path, bbox_weights, crnn_weights, output_viz=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Bounding Box Model
    bbox_model = BoundingBoxCNN(pretrained=False)
    if os.path.exists(bbox_weights):
        bbox_model.load_state_dict(torch.load(bbox_weights, map_location=device))
        bbox_model.eval()
        bbox_model.to(device)
    else:
        print(f"ERROR: BBox weights not found at {bbox_weights}")
        return
        
    # 2. Load CRNN Model
    crnn_model = CRNN().to(device)
    if os.path.exists(crnn_weights):
        crnn_model.load_state_dict(torch.load(crnn_weights, map_location=device, weights_only=True))
        crnn_model.eval()
        crnn_model.to(device)
    else:
        print(f"ERROR: CRNN weights not found at {crnn_weights}")
        return
        
    # Read the target image
    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found: {image_path}")
        return
        
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"ERROR: Failed to open image: {image_path}. Make sure it is a valid image file.")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    print(f"\nProcessing: {image_path}")
    
    # --- PHASE 1: BBox Detection ---
    print("-> Running Bounding Box Detection...")
    src_pts = get_bbox(img_rgb, bbox_model, device)
    
    # --- PHASE 2: Crop & Warp ---
    print("-> Extracting and Warping License Plate...")
    dst_pts = np.float32([[0, 0], [128, 0], [128, 32], [0, 32]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_plate = cv2.warpPerspective(img_bgr, M, (128, 32))
    
    # --- PHASE 3: CRNN Text Recognition ---
    print("-> Running CRNN Text Recognition...")
    predicted_text = read_text(warped_plate, crnn_model, device)
    
    print("\n" + "=" * 50)
    print(f" 🚘 PREDICTED LICENSE PLATE: {predicted_text} ")
    print("=" * 50 + "\n")
    
    # Save visual outputs
    if output_viz:
        # Draw bounding box polygon on the original high-res image
        pts = src_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_bgr, [pts], True, (0, 255, 0), 4)
        
        # Draw the predicted text immediately above the plate
        tl_pt = (int(src_pts[0][0]), int(src_pts[0][1] - 20))
        # Add a nice black background to make text readable
        (w, h), _ = cv2.getTextSize(predicted_text, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 5)
        cv2.rectangle(img_bgr, (tl_pt[0], tl_pt[1] - h - 10), (tl_pt[0] + w, tl_pt[1] + 10), (0, 0, 0), -1)
        cv2.putText(img_bgr, predicted_text, tl_pt, cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 5)
        
        cv2.imwrite(output_viz, img_bgr)
        print(f"Visualization saved to: {output_viz}")

        # Also save the 128x32 cropped plate for manual inspection
        crop_out = str(output_viz).replace(".jpg", "_crop.jpg")
        cv2.imwrite(crop_out, warped_plate)
        print(f"Cropped plate saved to: {crop_out}")

if __name__ == "__main__":
    # Default Paths dynamically resolved
    default_bbox = str(inference_dir / "bbox_model" / "weights" / "bbox_model.pth")
    default_crnn = str(inference_dir / "crnn_plate_epoch_50.pth")

    parser = argparse.ArgumentParser(description="Predict license plate string from a single arbitrary image.")
    parser.add_argument("image", type=str, help="Path to the query image")
    parser.add_argument("--bbox_weights", type=str, default=default_bbox, help="Path to BBox model weights")
    parser.add_argument("--crnn_weights", type=str, default=default_crnn, help="Path to CRNN model weights")
    parser.add_argument("--output", type=str, default="prediction_result.jpg", help="Path to save the visual result")
    
    args = parser.parse_args()
    
    predict_single_image(args.image, args.bbox_weights, args.crnn_weights, args.output)