# uv run inference_bbox.py PATH_TO_IMAGE.jpg --visualize

import torch
import cv2
import numpy as np
from aps360_lpr.train.src.bbox_model.bbox_cnn import BoundingBoxCNN
from torchvision import transforms
import argparse

def predict_bounding_box(image_path, model_path="weights/bbox_model.pth"):
    # Determine the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model = BoundingBoxCNN(pretrained=False)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
    except FileNotFoundError:
        print(f"Error: Model weights not found at {model_path}")
        return None

    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None
        
    orig_h, orig_w, _ = image.shape
    
    # Convert the image to low resolution first (e.g., width 800px, preserve aspect ratio)
    scale_ratio = 800.0 / orig_w
    low_res_w = int(orig_w * scale_ratio)
    low_res_h = int(orig_h * scale_ratio)
    low_res_img = cv2.resize(image, (low_res_w, low_res_h), interpolation=cv2.INTER_AREA)
    
    image_rgb = cv2.cvtColor(low_res_img, cv2.COLOR_BGR2RGB)
    
    # Needs to match the training resize and normalization
    resized_img = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(resized_img).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor).squeeze(0).cpu().numpy()
        
    # Denormalize coordinates to match our newly created low resolution image
    # Note: Expand the bounding box outward slightly (by a percentage margin)
    # The padding prevents edges of characters from being chopped off
    PADDING_PERCENT = 0.05
    
    points = []
    for i in range(0, 8, 2):
        # Base predicted coordinate
        px = output[i]
        py = output[i+1]
        
        # Calculate center
        center_x = (output[0] + output[2] + output[4] + output[6]) / 4.0
        center_y = (output[1] + output[3] + output[5] + output[7]) / 4.0
        
        # Expand away from center
        dx = px - center_x
        dy = py - center_y
        
        px_padded = px + (dx * PADDING_PERCENT)
        py_padded = py + (dy * PADDING_PERCENT)
        
        x = px_padded * low_res_w
        y = py_padded * low_res_h
        points.append((int(x), int(y)))
        
    return points, low_res_img

def draw_bounding_box(image, points, output_path="output_bbox.jpg"):
    # Draw points
    for i, pt in enumerate(points):
        cv2.circle(image, pt, 5, (0, 0, 255), -1)
        cv2.putText(image, str(i+1), (pt[0]+10, pt[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    # Draw polygon
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], True, (0, 255, 0), 3)
    
    cv2.imwrite(output_path, image)
    print(f"Saved prediction visualization to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict license plate bounding box")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--model", default="weights/bbox_model.pth", help="Path to model weights")
    parser.add_argument("--visualize", action="store_true", help="Draw bounding box and save to output.jpg")
    args = parser.parse_args()
    
    points, low_res_img = predict_bounding_box(args.image_path, args.model)
    
    if points:
        print("Predicted Coordinates (Scale adjusted to Low Res Image):")
        for i, pt in enumerate(points):
            print(f"Point {i+1}: {pt}")
            
        if args.visualize and low_res_img is not None:
            draw_bounding_box(low_res_img, points)