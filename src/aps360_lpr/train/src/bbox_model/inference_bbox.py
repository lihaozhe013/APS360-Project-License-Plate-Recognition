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
        return None
        
    orig_h, orig_w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Needs to match the training resize and normalization
    resized_img = cv2.resize(image_rgb, (224, 224))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(resized_img).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor).squeeze(0).cpu().numpy()
        
    # Denormalize coordinates (the model outputs [0, 1] range)
    points = []
    for i in range(0, 8, 2):
        x = output[i] * orig_w
        y = output[i+1] * orig_h
        points.append((int(x), int(y)))
        
    return points

def draw_bounding_box(image_path, points, output_path="output_bbox.jpg"):
    image = cv2.imread(image_path)
    
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
    
    points = predict_bounding_box(args.image_path, args.model)
    
    if points:
        print("Predicted Coordinates:")
        for i, pt in enumerate(points):
            print(f"Point {i+1}: {pt}")
            
        if args.visualize:
            draw_bounding_box(args.image_path, points)